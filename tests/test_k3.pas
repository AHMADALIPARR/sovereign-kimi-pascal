{ SPDX-License-Identifier: AGPL-3.0-only
  Pascal side of the k3_nano_harness.cu parity test: same inputs (same RNG sequence) as refs/k3_ref.cpp,
  but every kernel runs through SK_K3Kernels on pascal-stack (SK_Launch -> DispatchKernel). }
{$mode objfpc}{$H+}{$Q-}{$R-}{$POINTERMATH ON}
program test_k3;
uses SysUtils, Math, SKRng, SKOut, SK_Half, SK_K3Kernels, SK_K3Harness;
var r: TRng;
procedure Fill(var A: TSingleArray; N: Integer; S: Single); var i: Integer;
begin A := nil; SetLength(A, N); for i := 0 to N - 1 do A[i] := r.NextF * S; end;
procedure FillH(var A: THalfArray; N: Integer; S: Single); var i: Integer;
begin A := nil; SetLength(A, N); for i := 0 to N - 1 do A[i] := FloatToHalf(r.NextF * S); end;
var dim, i, j, rep, ne, k, inter, E, dp, ip, ot, wb, shape, V, G, a, n, step, idx, tok, p: Integer;
    inA, w, o, q, pk, ck, vv, rw, w1, w2, lg, TL, DL, te, pe, res: TSingleArray; inH, w1h, w2h, oh: THalfArray;
    sel, dt, acc, toks: TIntArray; seed: QWord; u: LongWord; f: Single; tokOut, cnt: LongInt;
    M: TK3Model; D: TDraftModel; Cfg: TK3Config; Wt: TK3Weights; logits: TSingleArray; nm: AnsiString; ids: array[0..2] of LongInt;
begin
  r := MakeRng(2026);
  dim := 300; Fill(inA, dim, 3); Fill(w, dim, 1); SetLength(o, dim);
  rmsnorm_kernel(1, 256, @inA[0], @w[0], @o[0], 1e-6, dim); for i := 0 to dim - 1 do OutF('rmsnorm', i, o[i]);
  dim := 96; Fill(inA, dim, 1); Fill(w, 3 * dim * dim, 1); o := nil; SetLength(o, 3 * dim);
  qkv_proj_kernel(DivCeil(3 * dim, 256), 256, @inA[0], @w[0], @o[0], dim); for i := 0 to 3 * dim - 1 do OutF('qkv_proj', i, o[i]);
  dim := 500; Fill(q, dim, 4); Fill(pk, dim, 4); Fill(ck, dim, 4); Fill(vv, dim, 1); o := nil; SetLength(o, dim);
  kimi_delta_attention_kernel(DivCeil(dim, 256), 256, @q[0], @pk[0], @ck[0], @vv[0], @o[0], 0.7, dim); for i := 0 to dim - 1 do OutF('delta_attn', i, o[i]);
  dim := 96; Fill(inA, dim, 1); Fill(w, dim * dim, 1); o := nil; SetLength(o, dim);
  attn_out_proj_kernel(DivCeil(dim, 256), 256, @inA[0], @w[0], @o[0], dim); for i := 0 to dim - 1 do OutF('attn_out_proj', i, o[i]);
  for rep := 0 to 4 do begin
    dim := 64; ne := 16; k := 4; Fill(inA, dim, 1); Fill(w, ne * dim, 0.5); rw := nil; SetLength(rw, k); sel := nil; SetLength(sel, k);
    latent_moe_router_kernel(1, 32, @inA[0], @w[0], @sel[0], @rw[0], dim, ne, k);
    for j := 0 to k - 1 do begin OutI('router_sel', rep * k + j, sel[j]); OutF('router_w', rep * k + j, rw[j]); end;
  end;
  dim := 40; inter := 70; E := 3; k := 2; Fill(inA, dim, 1); Fill(w1, E * dim * inter, 0.3); Fill(w2, E * inter * dim, 0.3); o := nil; SetLength(o, dim);
  sel := nil; SetLength(sel, 2); sel[0] := 2; sel[1] := 0; rw := nil; SetLength(rw, 2); rw[0] := 0.6; rw[1] := 0.4;
  warp_decode_moe_kernel(DivCeil(dim * 32, 128), 128, @inA[0], @w1[0], @w2[0], @sel[0], @rw[0], @o[0], dim, inter, k);
  for i := 0 to dim - 1 do OutF('warp_moe', i, o[i]);
  for shape := 0 to 1 do begin
    if shape = 1 then dim := 72 else dim := 40; inter := 70; E := 3; k := 2; dp := ((dim + 15) div 16) * 16; ip := ((inter + 15) div 16) * 16;
    FillH(inH, dp, 1); FillH(w1h, E * dp * ip, 0.3); FillH(w2h, E * ip * dp, 0.3);
    o := nil; SetLength(o, dim); for i := 0 to dim - 1 do o[i] := -7.0;
    sel[0] := 1; sel[1] := 2; rw[0] := 0.7; rw[1] := 0.3; ot := (dp + 15) div 16; wb := (ot * 32 + 127) div 128;
    wmma_decode_moe_kernel(wb, 128, @inH[0], @w1h[0], @w2h[0], @sel[0], @rw[0], @o[0], dim, inter, k, dp, ip);
    if shape = 1 then nm := 'wmma_moe_72' else nm := 'wmma_moe_40';
    for i := 0 to dim - 1 do OutF(nm, i, o[i]);
  end;
  dim := 37; dp := 48; Fill(inA, dim, 70000.0); inA[3] := 1e-6; inA[4] := 6.1e-5; inA[5] := -0.0;
  oh := nil; SetLength(oh, dp); for i := 0 to dp - 1 do oh[i] := FloatToHalf(1.0);
  convert_fp32_to_fp16_kernel(1, 256, @inA[0], @oh[0], dim, dp); for i := 0 to dp - 1 do OutI('fp32_to_fp16', i, oh[i]);
  for i := 0 to 3999 do begin u := LongWord(r.Next); f := PSingle(@u)^; if IsNan(f) then Continue; OutI('f2h_sweep', i, FloatToHalf(f)); end;
  V := 300; Fill(lg, V, 5); dim := 64; Fill(inA, dim, 1); Fill(w, V * dim, 1); o := nil; SetLength(o, V);
  output_head_kernel(DivCeil(V, 256), 256, @inA[0], @w[0], @o[0], V, dim); for i := 0 to V - 1 do OutF('output_head', i, o[i]);
  for rep := 0 to 19 do begin
    V := 300; Fill(lg, V, 4); seed := r.Next;
    top_p_sample_kernel(1, 1, @lg[0], V, 0.8, 0.9, seed, @tokOut); OutI('top_p_token', rep, tokOut); OutF('top_p_p0', rep, lg[0]);
  end;
  for rep := 0 to 19 do begin
    V := 200; G := 4; Fill(TL, (G + 1) * V, 3); Fill(DL, G * V, 3); dt := nil; SetLength(dt, 4); for i := 0 to 3 do dt[i] := LongInt(r.Next mod QWord(V));
    seed := r.Next; acc := nil; SetLength(acc, 4); for i := 0 to 3 do acc[i] := -1;
    speculative_acceptance_kernel(1, 1, @TL[0], @DL[0], @dt[0], V, G, 0.8, seed, @cnt, @acc[0]);
    OutI('spec_count', rep, cnt); for i := 0 to cnt - 1 do OutI('spec_tok', rep * 4 + i, acc[i]);
  end;
  dim := 50; Fill(te, 20 * dim, 1); Fill(pe, 10 * dim, 1); res := nil; SetLength(res, 3 * dim);
  ids[0] := 4; ids[1] := 19; ids[2] := 0;
  batched_embedding_kernel(3, 64, @ids[0], @te[0], @pe[0], @res[0], dim, 10, 2, 3);
  for i := 0 to 3 * dim - 1 do OutF('batched_embedding', i, res[i]);
  { full forward_token through the ported host harness }
  M := TK3Model.Create;
  LoadModelWeights('model_small.bin', Cfg, Wt, False); InitModel(M, Cfg); M.Weights := Wt; ConvertAndUploadFp16Weights(M, False);
  toks := nil; SetLength(toks, 8); TokenizePrompt('Hi!', toks, n, 8);
  logits := nil; SetLength(logits, Cfg.VocabSize);
  for i := 0 to n - 1 do ForwardToken(M, toks[i], i, @logits[0]);
  tok := 1; idx := 0;
  for step := 0 to 3 do begin
    ForwardToken(M, tok, n + step, @logits[0]);
    for p := 0 to Cfg.VocabSize - 1 do begin OutF('forward_logits', idx, logits[p]); Inc(idx); end;
    top_p_sample_kernel(1, 1, @M.Logits[0], Cfg.VocabSize, 0.8, 0.9, 1000 + step, @tokOut); tok := tokOut;
    OutI('forward_token', step, tok);
  end;
  D := TDraftModel.Create; LoadDraftModel('draft_small.bin', D, M);
  tok := 1; idx := 0;
  for p := 0 to 4 do begin
    tok := DraftStep(D, tok, p); OutI('draft_best', p, tok);
    for i := 0 to D.Config.VocabSize - 1 do begin OutF('draft_logits', idx, D.Logits[i]); Inc(idx); end;
  end;
  D.Free;
  FreeModel(M); M.Free;
end.
