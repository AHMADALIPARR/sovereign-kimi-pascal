{ SPDX-License-Identifier: AGPL-3.0-only — Pascal side of the Rust parity test }
{$mode objfpc}{$H+}{$Q-}{$R-}
program parity_rust_pas;
uses SysUtils, SKRng, SK_StateAudit, SK_ThreadPiper;
var r: TRng; n, i, g, q: Integer; x: LongWord; e: TAuditEvent; s, o: TPiperState;
    d: TDelta; p: TPiper; t: TTask;
begin
  n := 200; if ParamCount >= 1 then n := StrToInt(ParamStr(1));
  r := MakeRng(12345);
  for i := 1 to n do begin x := LongWord(r.Next); WriteLn('mix ', x, ' ', Mix(x)); end;
  for g := 1 to 420 do for i := 1 to n div 10 do begin x := LongWord(r.Next); WriteLn('gate ', g, ' ', x, ' ', RuntimeGate(g, x)); end;
  for g := 1 to 140 do for i := 1 to n div 10 do begin x := LongWord(r.Next); WriteLn('audit ', g, ' ', x, ' ', Audit(g, x)); end;
  for i := 1 to n do begin
    e.Node := LongWord(r.Next); e.Epoch := r.Next; e.Kind := Word(r.Next); e.State := LongWord(r.Next); e.Prev := LongWord(r.Next); e.Seal := 0;
    WriteLn('seal ', e.Node, ' ', e.Epoch, ' ', e.Kind, ' ', e.State, ' ', e.Prev, ' ', SealEvent(e));
  end;
  d := TDelta.Create;
  for i := 1 to n do begin
    s.Value := LongWord(r.Next); s.Seal := LongWord(r.Next); s.Epoch := r.Next shr 1; o := d.Map(s);
    WriteLn('delta ', s.Value, ' ', s.Seal, ' ', s.Epoch, ' ', o.Value, ' ', o.Seal, ' ', o.Epoch);
  end;
  d.Free;
  for q := 0 to 7 do p.Queues[q] := QueueEmpty;
  FillChar(p.Router.Weights, SizeOf(p.Router.Weights), 0); p.Router.Experts := 896;
  for i := 1 to n do begin x := LongWord(r.Next); WriteLn('route ', x, ' ', p.Route(x)); end;
  for q := 0 to 7 do begin
    for i := 0 to 63 do begin
      p.Queues[q].Slots[i].Id := LongWord(r.Next); p.Queues[q].Slots[i].Expert := Word(r.Next);
      p.Queues[q].Slots[i].Priority := Byte(r.Next); p.Queues[q].Slots[i].Ready := (r.Next and 1) = 1;
    end;
    p.Queues[q].Tail := r.Next mod 100;
  end;
  for i := 1 to n * 4 do begin
    q := r.Next mod 8;
    if p.Steal(q, t) then WriteLn('steal ', q, ' ', t.Id, ' ', t.Expert, ' ', t.Priority, ' ', Ord(t.Ready), ' ', p.Queues[q].Head)
    else WriteLn('steal ', q, ' none ', p.Queues[q].Head);
  end;
end.
