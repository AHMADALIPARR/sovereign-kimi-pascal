{ SPDX-License-Identifier: AGPL-3.0-only
  Port of src/ffi.rs (safe engine wrapper) and src/daemon.rs (channel actor pinned to one OS thread).
  src/lib.rs only re-exports these two modules; this unit is its equivalent.
  The engine entry points (k3_engine_init/forward/free, k3_nano_harness.cu:1124-1142) are bound
  through TK3EngineApi so the wrapper can drive the Pascal harness port (SK_K3Harness) or a stub. }
{$mode objfpc}{$H+}
unit SK_K3Engine;

interface

uses Classes, SysUtils, SyncObjs, Math;

type
  TSingleArray = array of Single;
  { ffi.rs:7-26 }
  TK3EngineApi = record
    Init: function(Path: PAnsiChar): Pointer;
    Forward: function(Engine: Pointer; Token, Pos: LongInt; OutLogits: PSingle): LongInt;
    Free: procedure(Engine: Pointer);
  end;

  EK3Engine = class(Exception);

  { ffi.rs:29-75 — single owner of the opaque engine pointer; Destroy == Drop }
  TK3Engine = class
  private
    FApi: TK3EngineApi; FPtr: Pointer; FVocab: PtrUInt;
  public
    constructor Create(const Api: TK3EngineApi; const ModelPath: AnsiString; VocabSize: PtrUInt); { ffi.rs:36-44 }
    destructor Destroy; override;                                                                  { ffi.rs:69-75 }
    function Forward(Token, Pos: LongInt): TSingleArray;                                          { ffi.rs:47-56 }
    function SampleGreedy(const Logits: array of Single): LongInt;                                { ffi.rs:59-66 }
  end;

  { daemon.rs:9-13 }
  TInferenceRequest = class
    Token, Pos: LongInt;
    Response: TSingleArray;
    Done: TEvent;          { oneshot channel }
    Failed: Boolean;
    constructor Create(AToken, APos: LongInt);
    destructor Destroy; override;
  end;

  TK3DaemonThread = class;

  { daemon.rs:15-74 }
  TK3DaemonHandle = class
  private
    FThread: TK3DaemonThread; FVocab: PtrUInt;
  public
    constructor Spawn(const Api: TK3EngineApi; const ModelPath: AnsiString; VocabSize: PtrUInt); { daemon.rs:26-58 }
    destructor Destroy; override;     { dropping the last Sender closes the channel -> engine freed }
    function Predict(Token, Pos: LongInt; out Logits: TSingleArray; out Err: AnsiString): Boolean; { daemon.rs:61-68 }
    function VocabSize: PtrUInt;      { daemon.rs:71-73 }
  end;

  { bounded mpsc channel (capacity 64, daemon.rs:27) + the dedicated OS thread (daemon.rs:31-55) }
  TK3DaemonThread = class(TThread)
  private
    FApi: TK3EngineApi; FPath: AnsiString; FVocab: PtrUInt;
    FLock: TCriticalSection; FNotEmpty, FNotFull: TEventObject;
    FItems: array[0..63] of TInferenceRequest; FHead, FCount: Integer;
    FClosed, FDead: Boolean;
  protected
    procedure Execute; override;
  public
    constructor Create(const Api: TK3EngineApi; const Path: AnsiString; Vocab: PtrUInt);
    destructor Destroy; override;
    function Send(R: TInferenceRequest): Boolean;
    procedure Close;
  end;

implementation

constructor TK3Engine.Create(const Api: TK3EngineApi; const ModelPath: AnsiString; VocabSize: PtrUInt);
begin
  inherited Create;
  if Pos(#0, ModelPath) > 0 then raise EK3Engine.Create('Invalid path: interior nul byte'); { CString::new }
  FApi := Api; FVocab := VocabSize;
  FPtr := FApi.Init(PAnsiChar(ModelPath));
  if FPtr = nil then raise EK3Engine.Create('Failed to initialize K3 engine');
end;

destructor TK3Engine.Destroy;
begin
  if FPtr <> nil then FApi.Free(FPtr);
  FPtr := nil;
  inherited Destroy;
end;

function TK3Engine.Forward(Token, Pos: LongInt): TSingleArray;
var Ret: LongInt;
begin
  Result := nil; SetLength(Result, FVocab);   { zero-initialised like vec![0.0; n] }
  if FVocab > 0 then Ret := FApi.Forward(FPtr, Token, Pos, @Result[0])
  else Ret := FApi.Forward(FPtr, Token, Pos, nil);
  if Ret <> 0 then raise EK3Engine.CreateFmt('Forward pass failed with code %d', [Ret]);
end;

{ Iterator::max_by returns the LAST maximal element on ties; partial_cmp(..).unwrap() panics on NaN. }
function TK3Engine.SampleGreedy(const Logits: array of Single): LongInt;
var i, Best: Integer;
begin
  if Length(Logits) = 0 then Exit(0);
  for i := 0 to High(Logits) do
    if IsNan(Logits[i]) then raise EK3Engine.Create('sample_greedy: NaN logit (Rust partial_cmp().unwrap() panics)');
  Best := 0;
  for i := 1 to High(Logits) do
    if Logits[i] >= Logits[Best] then Best := i;
  Result := Best;
end;

constructor TInferenceRequest.Create(AToken, APos: LongInt);
begin
  inherited Create; Token := AToken; Pos := APos; Done := TEvent.Create(nil, True, False, '');
end;

destructor TInferenceRequest.Destroy;
begin Done.Free; inherited Destroy; end;

constructor TK3DaemonThread.Create(const Api: TK3EngineApi; const Path: AnsiString; Vocab: PtrUInt);
begin
  FApi := Api; FPath := Path; FVocab := Vocab;
  FLock := TCriticalSection.Create;
  FNotEmpty := TEventObject.Create(nil, True, False, '');
  FNotFull := TEventObject.Create(nil, True, True, '');
  FreeOnTerminate := False;
  inherited Create(False);
end;

destructor TK3DaemonThread.Destroy;
begin
  FLock.Free; FNotEmpty.Free; FNotFull.Free;
  inherited Destroy;
end;

function TK3DaemonThread.Send(R: TInferenceRequest): Boolean;
begin
  repeat
    FLock.Enter;
    try
      if FClosed or FDead then Exit(False);
      if FCount < 64 then
      begin
        FItems[(FHead + FCount) mod 64] := R; Inc(FCount);
        FNotEmpty.SetEvent;
        if FCount = 64 then FNotFull.ResetEvent;
        Exit(True);
      end;
    finally
      FLock.Leave;
    end;
    FNotFull.WaitFor(INFINITE);
  until False;
end;

procedure TK3DaemonThread.Close;
begin
  FLock.Enter; FClosed := True; FNotEmpty.SetEvent; FLock.Leave;
end;

procedure TK3DaemonThread.Execute;
var Engine: Pointer; Buf: TSingleArray; R: TInferenceRequest; Got: Boolean; i: Integer;
begin
  Engine := FApi.Init(PAnsiChar(FPath));
  if Engine = nil then
  begin
    WriteLn(StdErr, '[DAEMON] Failed to init CUDA engine');
    FLock.Enter; FDead := True;
    for i := 0 to FCount - 1 do begin FItems[(FHead + i) mod 64].Failed := True; FItems[(FHead + i) mod 64].Done.SetEvent; end;
    FCount := 0; FLock.Leave;
    Exit;
  end;
  Buf := nil; SetLength(Buf, FVocab);
  while True do   { while let Some(req) = rx.blocking_recv() }
  begin
    FNotEmpty.WaitFor(INFINITE);
    FLock.Enter;
    Got := FCount > 0;
    if Got then
    begin
      R := FItems[FHead]; FHead := (FHead + 1) mod 64; Dec(FCount);
      FNotFull.SetEvent;
      if FCount = 0 then FNotEmpty.ResetEvent;
    end;
    if (not Got) and FClosed then begin FLock.Leave; Break; end;
    FLock.Leave;
    if not Got then Continue;
    if FVocab > 0 then FApi.Forward(Engine, R.Token, R.Pos, @Buf[0])   { return code ignored, daemon.rs:43-48 }
    else FApi.Forward(Engine, R.Token, R.Pos, nil);
    R.Response := Copy(Buf, 0, Length(Buf));   { logits_buf.clone() }
    R.Done.SetEvent;
  end;
  FApi.Free(Engine);
  WriteLn(StdErr, '[DAEMON] CUDA engine freed');
end;

constructor TK3DaemonHandle.Spawn(const Api: TK3EngineApi; const ModelPath: AnsiString; VocabSize: PtrUInt);
begin
  inherited Create;
  if Pos(#0, ModelPath) > 0 then raise EK3Engine.Create('Invalid CString path');  { daemon.rs:28 expect() }
  FVocab := VocabSize;
  FThread := TK3DaemonThread.Create(Api, ModelPath, VocabSize);
end;

destructor TK3DaemonHandle.Destroy;
begin
  if FThread <> nil then begin FThread.Close; FThread.WaitFor; FThread.Free; end;
  inherited Destroy;
end;

function TK3DaemonHandle.Predict(Token, Pos: LongInt; out Logits: TSingleArray; out Err: AnsiString): Boolean;
var R: TInferenceRequest;
begin
  Logits := nil; Err := '';
  R := TInferenceRequest.Create(Token, Pos);
  try
    if not FThread.Send(R) then begin Err := 'Channel send failed: channel closed'; Exit(False); end;
    R.Done.WaitFor(INFINITE);
    if R.Failed then begin Err := 'Channel recv failed: channel closed'; Exit(False); end;
    Logits := R.Response; Result := True;
  finally
    R.Free;
  end;
end;

function TK3DaemonHandle.VocabSize: PtrUInt;
begin Result := FVocab; end;

end.
