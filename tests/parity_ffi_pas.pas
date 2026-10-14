{ SPDX-License-Identifier: AGPL-3.0-only — Pascal side of the ffi.rs/daemon.rs parity test (same stub engine) }
{$mode objfpc}{$H+}{$Q-}{$R-}
program parity_ffi_pas;
uses {$ifdef unix}cthreads,{$endif} SysUtils, SK_K3Engine;
const VOCAB = 300;
type PStub = ^TStub; TStub = record Calls: LongInt; end;
function StubInit(P: PAnsiChar): Pointer; var S: PStub; begin New(S); S^.Calls := 0; Result := S; end;
function StubForward(E: Pointer; Token, Pos: LongInt; OutL: PSingle): LongInt;
var i: Integer; h: LongWord;
begin
  Inc(PStub(E)^.Calls);
  for i := 0 to VOCAB - 1 do begin
    h := (LongWord(Token) * LongWord(2654435761)) xor (LongWord(Pos) * LongWord(40503)) xor (LongWord(i) * LongWord(2246822519));
    h := h xor (h shr 15); h := h * LongWord($2c1b3c6d); h := h xor (h shr 12);
    OutL[i] := (Single(h mod 20001) - 10000.0) / 100.0;
  end;
  Result := 0;
end;
procedure StubFree(E: Pointer); begin Dispose(PStub(E)); end;
var Api: TK3EngineApi; Eng: TK3Engine; H: TK3DaemonHandle; t, p: Integer; L: TSingleArray; Err: AnsiString; s: Double; i: Integer;
    Empty: array of Single;
begin
  Api.Init := @StubInit; Api.Forward := @StubForward; Api.Free := @StubFree;
  Eng := TK3Engine.Create(Api, 'model.bin', VOCAB);
  for t := 0 to 49 do for p := 0 to 3 do begin L := Eng.Forward(t * 7 + 1, p); WriteLn('greedy ', t * 7 + 1, ' ', p, ' ', Eng.SampleGreedy(L)); end;
  Empty := nil;
  WriteLn('greedy_empty ', Eng.SampleGreedy(Empty));
  WriteLn('greedy_tie ', Eng.SampleGreedy([1.0, 3.0, 3.0, 2.0]));
  H := TK3DaemonHandle.Spawn(Api, 'model.bin', VOCAB);
  WriteLn('vocab ', H.VocabSize);
  for t := 0 to 39 do begin
    if not H.Predict(t, t * 3, L, Err) then begin WriteLn('ERR ', Err); Halt(1); end;
    s := 0; for i := 0 to High(L) do s := s + L[i];
    WriteLn('predict ', t, ' ', t * 3, ' len=', Length(L), ' sum=', FormatFloat('0.0000', s), ' first=', FormatFloat('0.00', L[0]));
  end;
  H.Free; Eng.Free;
end.
