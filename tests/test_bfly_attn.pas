{ SPDX-License-Identifier: AGPL-3.0-only
  Pascal side of the butterfly / SASS / attention parity test (same RNG sequence as refs/bfly_attn_ref.cpp). }
{$mode objfpc}{$H+}{$Q-}{$R-}{$POINTERMATH ON}
program test_bfly_attn;
uses SysUtils, Math, SKRng, SKOut, SK_Half, SK_FloatOps, SK_Butterfly, SK_Attention;
var r: TRng;
  dim, i, l, t, w, st, n, q, k, j, tmpI: Integer;
  x, rw, p, g, s, stt, O: array of Single; ids: array[0..15] of LongInt; wt: array[0..15] of Single;
  seal, outU, mem: array of LongWord; perm: array of LongInt;
  v, v2, v3: TWarpF; a, b, c, fx, fy: Single; ux, uy, xx, yy, zz: LongWord;
  regs: array of TSassWarp; Qh, Kh, Vh: array of THalf;
const QS: array[0..4] of Integer = (6, 7, 8, 9, 11); QO: array[0..5] of Integer = (0, 1, 2, 8, 12, 13);
begin
  r := MakeRng(4242);
  { T1 }
  dim := 100; SetLength(x, dim); SetLength(rw, 896 * dim);
  for i := 0 to dim - 1 do x[i] := r.NextF; for i := 0 to 896 * dim - 1 do rw[i] := r.NextF;
  threadpiper_router(1, 896, @x[0], @rw[0], @ids[0], @wt[0], dim);
  for i := 0 to 15 do begin OutI('tp_router_ids', i, ids[i]); OutF('tp_router_w', i, wt[i]); end;
  { T2 }
  dim := 256; SetLength(x, dim); SetLength(p, dim); SetLength(g, dim); SetLength(stt, dim);
  for i := 0 to dim - 1 do x[i] := r.NextF; for i := 0 to dim - 1 do p[i] := r.NextF; for i := 0 to dim - 1 do g[i] := r.NextF;
  for i := 0 to 15 do begin ids[i] := LongInt(r.Next mod 16); wt[i] := r.NextF; end;
  gated_delta_butterfly(2, 128, @x[0], @p[0], @g[0], @ids[0], @wt[0], @stt[0], dim);
  for i := 0 to dim - 1 do OutF('gated_delta', i, stt[i]);
  { T3 }
  n := 1000; SetLength(s, n); SetLength(seal, n); SetLength(outU, n);
  for i := 0 to n - 1 do begin s[i] := r.NextF * 1000; seal[i] := LongWord(r.Next); end;
  s[0] := -0.0; s[1] := Infinity; s[2] := 1e-40;
  triple_lock_kernel(4, 256, @s[0], @seal[0], @outU[0], n);
  for i := 0 to n - 1 do OutI('triple_lock', i, outU[i]);
  { T4 }
  for t := 0 to 39 do begin
    for l := 0 to 31 do begin v[l] := r.NextF * 10; v2[l] := v[l]; v3[l] := v[l]; end;
    FactorBfly(v); Reduce16(v2); st := r.Next mod 7; RecursiveReduce(v3, st);
    for l := 0 to 31 do begin OutF('factor_bfly', t * 32 + l, v[l]); OutF('reduce16', t * 32 + l, v2[l]); OutF('recursive_reduce', t * 32 + l, v3[l]); end;
  end;
  for t := 0 to 199 do begin
    a := r.NextF * 5; b := r.NextF * 5; c := r.NextF * 5; xx := LongWord(r.Next); yy := LongWord(r.Next); zz := LongWord(r.Next);
    OutF('factor_gate', t, FactorGate(a, b)); OutF('factor_delta', t, FactorDelta(a, b)); OutF('factor_recur', t, FactorRecur(c, a, b));
    OutI('factor_lock', t, FactorLock(xx, yy, zz)); OutF('route', t, Route(a, b));
  end;
  for t := 0 to 19999 do begin
    ux := LongWord(r.Next); uy := LongWord(r.Next);
    fx := UIntAsFloat((ux and $807FFFFF) or (LongWord(100 + (ux shr 23) mod 56) shl 23));
    fy := UIntAsFloat((uy and $807FFFFF) or (LongWord(100 + (uy shr 23) mod 56) shl 23));
    if t mod 4 = 0 then fy := -1.0 + UIntAsFloat((uy and $0000FFFF) or $33800000);
    OutF('factor_fmaf', t, Factor(1 + t mod FACTOR_COUNT, fx, fy));
  end;
  { T5 }
  SetLength(regs, 4); FillChar(regs[0], 4 * SizeOf(TSassWarp), 0);
  SetLength(mem, 256); for i := 0 to 255 do mem[i] := LongWord(r.Next);
  SetLength(perm, 256); for i := 0 to 255 do perm[i] := i;
  for i := 255 downto 1 do begin j := LongInt(r.Next mod QWord(i + 1)); tmpI := perm[i]; perm[i] := perm[j]; perm[j] := tmpI; end;
  for w := 0 to 3 do for l := 0 to 31 do begin
    for q := 0 to 4 do regs[w][l].R[QS[q]] := FloatAsUInt(r.NextF * 8);
    regs[w][l].R[14] := LongWord(r.Next); regs[w][l].R[15] := LongWord(r.Next);
    regs[w][l].R[20] := LongWord(r.Next mod 256); regs[w][l].R[21] := perm[w * 32 + l];
  end;
  butterfly_sass_kernel(2, 64, @regs[0], @mem[0]);
  k := 0;
  for w := 0 to 3 do for l := 0 to 31 do begin
    for q := 0 to 5 do begin OutI('sass_int', k, regs[w][l].R[QO[q]]); Inc(k); end;
    OutF('sass_R9', w * 32 + l, UIntAsFloat(regs[w][l].R[9])); OutF('sass_R10', w * 32 + l, UIntAsFloat(regs[w][l].R[10]));
  end;
  for i := 0 to 255 do OutI('sass_mem', i, mem[i]);
  { T6 fused_attention }
  n := 256 * 64; SetLength(Qh, n); SetLength(Kh, n); SetLength(Vh, n); SetLength(O, n);
  for i := 0 to n - 1 do Qh[i] := FloatToHalf(r.NextF * 2); for i := 0 to n - 1 do Kh[i] := FloatToHalf(r.NextF * 2);
  for i := 0 to n - 1 do O[i] := -1.0;
  launch_fused_attention(@Qh[0], @Kh[0], @Vh[0], @O[0], 256, 64);
  for i := 0 to n - 1 do OutF('fused_attention', i, O[i]);
  { T7 gemm_online_softmax }
  n := 200 * 64; SetLength(Qh, n); SetLength(Kh, n); SetLength(Vh, n); SetLength(O, n);
  for i := 0 to n - 1 do Qh[i] := FloatToHalf(r.NextF); for i := 0 to n - 1 do Kh[i] := FloatToHalf(r.NextF); for i := 0 to n - 1 do Vh[i] := FloatToHalf(r.NextF);
  for i := 0 to n - 1 do O[i] := -1.0;
  launch_gemm_online_softmax(@Qh[0], @Kh[0], @Vh[0], @O[0], 200, 64);
  for i := 0 to n - 1 do OutF('gemm_online_softmax', i, O[i]);
end.
