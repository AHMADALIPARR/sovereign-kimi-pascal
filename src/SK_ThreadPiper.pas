{ SPDX-License-Identifier: AGPL-3.0-only
  Port of kimi_butterfly/runtime/threadpiper.rs (no_std ThreadPiper scheduler). }
{$mode objfpc}{$H+}{$modeswitch advancedrecords}{$Q-}{$R-}
unit SK_ThreadPiper;

interface

type
  { threadpiper.rs:3 }
  TTask = record Id: LongWord; Expert: Word; Priority: Byte; Ready: Boolean; end;
  { threadpiper.rs:4 }
  TPiperState = record Value: LongWord; Seal: LongWord; Epoch: QWord; end;
  { threadpiper.rs:5 }
  TQueue = record Slots: array[0..63] of TTask; Head, Tail: PtrUInt; end;
  { threadpiper.rs:6-7 — Functor trait; Delta is its only impl }
  generic IFunctor<TIn, TOut> = interface
    function Map(const X: TIn): TOut;
  end;
  TDelta = class(TInterfacedObject, specialize IFunctor<TPiperState, TPiperState>)
    function Map(const X: TPiperState): TPiperState;
  end;
  { threadpiper.rs:8 }
  TRouter = record Weights: array[0..15] of LongWord; Experts: Word; end;
  { threadpiper.rs:9,11 }
  TPiper = record
    Queues: array[0..7] of TQueue;
    Router: TRouter;
    function Steal(From: PtrUInt; out T: TTask): Boolean;   { Option<Task> }
    function Route(X: LongWord): Word;
  end;

function QueueEmpty: TQueue;                      { threadpiper.rs:10 }
function RuntimeGate(N: Integer; X: LongWord): LongWord;  { threadpiper.rs:12-431 }

implementation

{$I threadpiper_table.inc}

function TDelta.Map(const X: TPiperState): TPiperState;
begin
  Result.Value := X.Value xor LongWord($9e3779b9);
  Result.Seal := X.Seal;
  Result.Epoch := X.Epoch + 1;
end;

function QueueEmpty: TQueue;
var i: Integer;
begin
  for i := 0 to 63 do
  begin
    Result.Slots[i].Id := 0; Result.Slots[i].Expert := 0;
    Result.Slots[i].Priority := 0; Result.Slots[i].Ready := False;
  end;
  Result.Head := 0; Result.Tail := 0;
end;

function TPiper.Steal(From: PtrUInt; out T: TTask): Boolean;
var i: PtrUInt;
begin
  Assert(From < 8, 'Piper.steal: index out of bounds (Rust would panic)');
  if Queues[From].Head = Queues[From].Tail then Exit(False);
  i := Queues[From].Head;
  Inc(Queues[From].Head);
  T := Queues[From].Slots[i mod 64];
  Result := True;
end;

function TPiper.Route(X: LongWord): Word;
begin
  Result := Word(X mod 896);
end;

function RuntimeGate(N: Integer; X: LongWord): LongWord;
begin
  Assert((N >= 1) and (N <= 420), 'runtime_gate index out of range (001..420)');
  Result := RolDWord(X, GATE_ROT[N]) xor LongWord($A5A5A5A5);
end;

end.
