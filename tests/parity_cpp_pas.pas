{ SPDX-License-Identifier: AGPL-3.0-only — Pascal side of the C++ parity test }
{$mode objfpc}{$H+}{$Q-}{$R-}
program parity_cpp_pas;
uses SysUtils, SKRng, SK_KernelSpace, SK_ForgeCompiler;
var r: TRng; n, k, i, t, nn, na, pidx: Integer; x, a, h, hh, hp: QWord; need: Byte;
    pt: TPageTable; c1, c2, c3: TKContext; c: TForgeContext; ast: TAst; ir: TIR;
    p, s: AnsiString; j: SizeInt;
procedure Fnv(var H: QWord; V: QWord); begin H := (H xor V) * QWord(1099511628211); end;
begin
  n := 200; if ParamCount >= 1 then n := StrToInt(ParamStr(1));
  r := MakeRng(777);
  for k := 1 to 825 do for i := 1 to n div 20 do begin x := r.Next; WriteLn('kpath ', k, ' ', x, ' ', KernelPath(k, x)); end;
  for i := 0 to n - 1 do begin a := r.Next; if i mod 3 = 0 then a := a and not QWord($FFF);
    WriteLn('valid ', a, ' ', Ord(Valid(a)), ' ', Vpn(a)); end;
  for i := 0 to 511 do begin pt.E[i].Frame := r.Next; pt.E[i].Perm := Byte(r.Next and 7); pt.E[i].Cache := 0; pt.E[i].Asid := 0; pt.E[i].Flags := 0; end;
  for i := 0 to n * 4 - 1 do begin a := r.Next; need := Byte(r.Next and 7); WriteLn('xlate ', a, ' ', need, ' ', Translate(pt, a, need)); end;
  FillChar(c1, SizeOf(c1), 0); FillChar(c2, SizeOf(c2), 0); FillChar(c3, SizeOf(c3), 0);
  for i := 0 to 63 do c1.Gpr[i] := r.Next; c1.Pc := r.Next; c1.Sp := r.Next; c1.Status := r.Next; c1.Asid := LongWord(r.Next);
  Save(c2, c1); Restore(c3, c2); h := 0; for i := 0 to 63 do h := h xor (c3.Gpr[i] * QWord(i + 1));
  WriteLn('ctx ', h, ' ', c3.Pc, ' ', c3.Sp, ' ', c3.Status, ' ', c3.Asid);
  for t := 0 to n - 1 do
  begin
    FillChar(c.Target, SizeOf(c.Target), 0); c.Errors := nil; c.Symbols := nil;
    c.Target.Sm := r.Next mod 100; c.Target.RegsPerThread := r.Next mod 300;
    if r.Next mod 4 = 0 then c.Target.Warp := 64 else c.Target.Warp := 32;
    ast.Nodes := nil; ast.Root := 0; nn := r.Next mod 300; SetLength(ast.Nodes, nn);
    for i := 0 to nn - 1 do begin
      ast.Nodes[i].Kind := TKind(r.Next mod 16); ast.Nodes[i].Imm := Int64(r.Next);
      na := r.Next mod 5; if r.Next mod 50 = 0 then na := 70000; SetLength(ast.Nodes[i].Args, na);
    end;
    ir := Compile(c, ast);
    pidx := r.Next mod 1490; RunPass(pidx + 1, c, ir, ast);
    p := EmitP45(ir); s := EmitSass(ir);
    hh := QWord(1469598103934665603);
    for j := 0 to High(ir.Code) do begin Fnv(hh, ir.Code[j].Op); Fnv(hh, ir.Code[j].Dst); Fnv(hh, ir.Code[j].A); Fnv(hh, ir.Code[j].B); Fnv(hh, LongWord(ir.Code[j].Imm)); end;
    hp := QWord(1469598103934665603);
    for j := 1 to Length(p) do Fnv(hp, Ord(p[j])); for j := 1 to Length(s) do Fnv(hp, Ord(s[j]));
    Write('forge ', t, ' regs=', ir.Regs, ' code=', Length(ir.Code), ' ih=', hh, ' eh=', hp, ' plen=', Length(p), ' slen=', Length(s), ' errs=', Length(c.Errors));
    for j := 0 to High(c.Errors) do Write(' [', c.Errors[j].Text, ']'); WriteLn;
    if t = 0 then WriteLn('p45first ', Copy(p, 1, 60));
  end;
end.
