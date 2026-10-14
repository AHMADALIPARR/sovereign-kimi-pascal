{ SPDX-License-Identifier: AGPL-3.0-only
  Self-checks with no runnable original: forge.mlw contracts (Why3 is not installed, so the .mlw was
  NOT machine-checked here), microcode / P4.5 ISA loaders, RTL multiply model. }
{$mode objfpc}{$H+}{$ASSERTIONS ON}
program test_selfchecks;
uses SysUtils, SK_ForgeProof, SK_TensorPkg;
var pass, fail: Integer;
procedure Check(Cond: Boolean; const What: AnsiString);
begin if Cond then Inc(pass) else begin Inc(fail); WriteLn('FAIL ', What); end; end;
var x, n: Int64; i, bad: Integer; caught: Boolean; mc: TMicrocode; isa: TP45Isa;
begin
  pass := 0; fail := 0;
  for x := -50 to 50 do for n := -2 to 20 do
    if n <= 0 then Check(Butterfly(x, n) = x, 'butterfly base') else Check(Butterfly(x, n) = x * (Int64(1) shl n), 'butterfly = x*2^n');
  for x := 0 to 100000 do CheckButterflyNonnegative(x);
  Inc(pass);
  Check(RegOk(0, 256) and RegOk(255, 256) and not RegOk(256, 256) and not RegOk(-1, 256), 'reg_ok');
  Check(Aligned(0) and Aligned(4096) and Aligned(-4096) and not Aligned(4095) and not Aligned(-1), 'aligned (Euclidean mod)');
  Check(Recur(3, 4, 5) = 23, 'recur'); Check(Gate(-3, 7) = -21, 'gate'); Check(Delta(2, 9) = -7, 'delta');
  caught := False;
  try Butterfly(High(Int64) div 2 + 1, 1); except on E: EIntOverflow do caught := True; end;
  Check(caught, 'Int64 overflow outside Why3 int domain is trapped');
  mc := LoadMicrocode('../microcode/extended_ws_os_tc.txt');
  Check(Length(mc) = 25, 'microcode: 25 words parsed');
  bad := 0;
  for i := 0 to High(mc) do if not mc[i].Consistent then begin Inc(bad);
    WriteLn(Format('microcode line %d addr %.4x: hex %.8x <> binary %.8x', [mc[i].Line, mc[i].Addr, mc[i].Word, mc[i].BinWord])); end;
  WriteLn('microcode: ', bad, ' of ', Length(mc), ' lines have hex <> binary (data defect in the original file)');
  Check((mc[0].Addr = $0800) and (mc[24].Addr = $0826) and (mc[24].Word = $11110001), 'microcode first/last');
  isa := LoadP45Isa('../microcode/p45.isa');
  Check((isa.Name = 'P45.1') and (isa.Regs = 256) and (isa.Warp = 32) and (isa.Experts = 896) and (isa.Active = 16), 'p45 header');
  Check((Length(isa.Ops) = 20) and (isa.Ops[0] = 'BF_XOR') and (isa.Ops[19] = 'CACHE_FREEZE'), 'p45 ops');
  Check(isa.RuleCount = 280, 'p45 rules 1..280 contiguous');
  Check(Fp16MulFp32Rtl($FFFF, $0002) = LongWord(-2), 'RTL signed multiply');
  Check(Fp16MulFp32Rtl($3C00, $3C00) = LongWord(15360 * 15360), 'RTL multiply is integer, not FP16');
  WriteLn('selfchecks: ', pass, ' passed, ', fail, ' failed');
  if fail > 0 then Halt(1);
end.
