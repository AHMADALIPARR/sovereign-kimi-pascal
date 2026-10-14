{ SPDX-License-Identifier: AGPL-3.0-only
  Ports onto pascal-stack (block granularity, see SK_Launch) of
   * kernels/fused_attention.cu      gemm_online_softmax_kernel (:122-228) + launch_fused_attention (:234-260)
   * kernels/gemm_online_softmax.cu  gemm_online_softmax_kernel (:122-265) + launch_gemm_online_softmax (:271-295)
  Both kernels are reproduced literally, including their defects (see PORT_NOTES.md). Where the CUDA
  source has no defined behaviour, the port makes ONE explicit, documented choice:
   (a) dynamic/static shared memory starts zeroed;
   (b) cp.async copies land at the matching cp.async.wait (not earlier), so the compute that is issued
       between the copy and the wait reads the previous shared-memory contents;
   (c) in gemm_online_softmax.cu, reads of row_max[r]/row_denom[r] with r >= 32 (outside the
       32-entry per-thread arrays, :147-148, read at :247/:250 for warps 1..3) return NaN;
   (d) O_acc (:157, never initialised) starts at 0. }
{$mode objfpc}{$H+}{$Q-}{$R-}{$POINTERMATH ON}
unit SK_Attention;

interface

uses SysUtils, Math, PascalGPU_Types, SK_Launch, SK_Half;

const
  FA_TILE_M = 128; FA_TILE_N = 64; FA_TILE_K = 64; FA_Q_PADDING = 8; FA_K_PADDING = 8;   { fused_attention.cu:38-48 }
  FA_STABILITY_FACTOR: Single = 1.0 / 64.0;                                               { :50 }
  GO_TILE_M = 128; GO_TILE_N = 64; GO_TILE_K = 64; GO_WARPS = 4; GO_THREADS = 128; GO_Q_PAD = 8; GO_K_PAD = 8; { gemm_online_softmax.cu:32-38 }

{ fused_attention.cu. Output is addressed as float2 (:222) although declared half*: OutputF must hold
  seq_len*dim_head floats (the original half buffer of seq*dim halves is overrun for rows >= seq/2). }
procedure launch_fused_attention(Q, K, V: PHalf; OutputF: PSingle; SeqLen, DimHead: LongInt);
function FusedAttentionSmemBytes: LongWord;   { :243-246 }

{ gemm_online_softmax.cu }
procedure launch_gemm_online_softmax(Q, K, V: PHalf; O: PSingle; SeqLen, HeadDim: LongInt);
function GemmOnlineSoftmaxSmemBytes(HeadDim: LongInt): LongWord;   { :277-281 }

implementation

{ ===================== fused_attention.cu ===================== }
type
  TFaArgs = record Q, K, V: PHalf; OutF: PSingle; Seq, Dim: LongInt; end;
  PFaArgs = ^TFaArgs;

function FusedAttentionSmemBytes: LongWord;
begin
  Result := (FA_TILE_M * (FA_TILE_K + FA_Q_PADDING) + FA_TILE_N * (FA_TILE_K + FA_K_PADDING)) * SizeOf(THalf)
            + (FA_TILE_M * FA_TILE_N) * SizeOf(Single);
end;

{ online_softmax_step (:103-116) }
procedure OnlineStep(var M, D: Single; Score, SF: Single); inline;
var mPrev, ePrev, eCur: Single;
begin
  mPrev := M;
  if Score > M then M := Score;
  ePrev := Single(Exp((mPrev - M) * SF));
  eCur := Single(Exp((Score - M) * SF));
  D := D * ePrev + eCur;
end;

procedure FaBody(const T, B, BD, GD: TDim3D; Data: Pointer);
const NT = 128; ROWQ = FA_TILE_K + FA_Q_PADDING; ROWK = FA_TILE_K + FA_K_PADDING;
var A: PFaArgs; smem: array of Byte; sQ, sK: PHalf; bx, tx, warp, lane, numK, kStep, nextK, m, n, f, g, tg, l, r, c, kk: LongInt;
    mi, di: array[0..NT - 1] of Single; fc: array[0..NT - 1, 0..3] of Single;
    pend: array[0..NT - 1] of record Dst: SizeInt; Src: PHalf; end; npend: Integer;
    Am: array[0..15, 0..15] of Single; Bm: array[0..15, 0..7] of Single; Dm: array[0..15, 0..7] of Single;
    aH, bH: PHalf; s, blockMax: Single; red: array[0..31] of Single; tmp: array[0..31] of Single; off: Integer;
    outRow, outCol: LongInt; idx: SizeInt; qRow, qCol: LongInt;
begin
  A := PFaArgs(Data);
  smem := nil; SetLength(smem, FusedAttentionSmemBytes);          { (a) zeroed }
  sQ := PHalf(@smem[0]);
  sK := sQ + FA_TILE_M * ROWQ;
  bx := B.X;
  for tx := 0 to NT - 1 do begin mi[tx] := NegInfinity; di[tx] := 0.0; for f := 0 to 3 do fc[tx, f] := 0.0; end;
  numK := A^.Seq div FA_TILE_K;
  { prologue (:153-162): 16-byte cp.async per thread, waited immediately }
  for tx := 0 to NT - 1 do
  begin
    qRow := bx * FA_TILE_M + (tx div (FA_TILE_K div 8));
    qCol := (tx mod (FA_TILE_K div 8)) * 8;
    Move(A^.Q[SizeInt(qRow) * A^.Dim + qCol], sQ[(tx div (FA_TILE_K div 8)) * ROWQ + qCol], 16);
  end;
  for kStep := 0 to numK - 1 do
  begin
    nextK := kStep + 1;
    npend := 0;
    if nextK < numK then
      for tx := 0 to NT - 1 do   { (b) queued, lands at wait_group0 }
      begin
        pend[npend].Dst := (tx div (FA_TILE_N div 8)) * ROWK + (tx mod (FA_TILE_N div 8)) * 8;
        pend[npend].Src := @A^.K[SizeInt(nextK * FA_TILE_K + (tx div (FA_TILE_N div 8))) * A^.Dim + (tx mod (FA_TILE_N div 8)) * 8];
        Inc(npend);
      end;
    m := 0;
    while m < FA_TILE_M do
    begin
      n := 0;
      while n < FA_TILE_N do
      begin
        for warp := 0 to 3 do
        begin
          { A fragment: all lanes load the same 4 regs from smem_Q[(m + warp*16) * ROWQ] (:182-190) }
          aH := sQ + (m + warp * 16) * ROWQ;
          { PTX m16n8k16 .row A layout: a0={A[g][2t],A[g][2t+1]} a1={A[g+8][..]} a2={A[g][2t+8],..} a3={A[g+8][2t+8],..} }
          for r := 0 to 15 do for c := 0 to 15 do
            Am[r, c] := HalfToFloat(aH[((Ord(r >= 8)) + 2 * Ord(c >= 8)) * 2 + (c and 1)]);
          { B fragment: lane L loads 2 regs from smem_K[(n + L) * ROWK]; b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]} }
          for l := 0 to 31 do
          begin
            g := l shr 2; tg := l and 3;
            bH := sK + (n + l) * ROWK;
            Bm[2 * tg, g] := HalfToFloat(bH[0]); Bm[2 * tg + 1, g] := HalfToFloat(bH[1]);
            Bm[2 * tg + 8, g] := HalfToFloat(bH[2]); Bm[2 * tg + 9, g] := HalfToFloat(bH[3]);
          end;
          { C/D layout: c0=D[g][2t] c1=D[g][2t+1] c2=D[g+8][2t] c3=D[g+8][2t+1] }
          for l := 0 to 31 do
          begin
            g := l shr 2; tg := l and 3; tx := warp * 32 + l;
            Dm[g, 2 * tg] := fc[tx, 0]; Dm[g, 2 * tg + 1] := fc[tx, 1];
            Dm[g + 8, 2 * tg] := fc[tx, 2]; Dm[g + 8, 2 * tg + 1] := fc[tx, 3];
          end;
          for r := 0 to 15 do for c := 0 to 7 do
          begin
            s := Dm[r, c];
            for kk := 0 to 15 do s := s + Am[r, kk] * Bm[kk, c];
            Dm[r, c] := s;
          end;
          for l := 0 to 31 do
          begin
            g := l shr 2; tg := l and 3; tx := warp * 32 + l;
            fc[tx, 0] := Dm[g, 2 * tg]; fc[tx, 1] := Dm[g, 2 * tg + 1];
            fc[tx, 2] := Dm[g + 8, 2 * tg]; fc[tx, 3] := Dm[g + 8, 2 * tg + 1];
            for f := 0 to 3 do OnlineStep(mi[tx], di[tx], fc[tx, f], FA_STABILITY_FACTOR);
          end;
        end;
        Inc(n, 8);
      end;
      Inc(m, 16);
    end;
    if nextK < numK then
      for f := 0 to npend - 1 do Move(pend[f].Src^, sK[pend[f].Dst], 16);   { cp.async.wait_group 0 }
  end;
  { epilogue (:210-227) }
  for warp := 0 to 3 do
  begin
    for l := 0 to 31 do red[l] := mi[warp * 32 + l];
    off := 16;
    while off >= 1 do
    begin
      for l := 0 to 31 do tmp[l] := red[l xor off];
      for l := 0 to 31 do if not (red[l] > tmp[l]) then red[l] := tmp[l];   { (val > other) ? val : other }
      off := off div 2;
    end;
    for l := 0 to 31 do
    begin
      tx := warp * 32 + l; blockMax := red[l];
      for f := 0 to 3 do fc[tx, f] := Single(Exp((fc[tx, f] - blockMax) * FA_STABILITY_FACTOR)) / di[tx];
      outRow := bx * FA_TILE_M + warp * 16 + l div 4;
      outCol := (l mod 4) * 2;
      if (outRow < A^.Seq) and (outCol < A^.Dim) then
      begin
        idx := (SizeInt(outRow) * (A^.Dim div 2) + (outCol div 2)) * 2;
        A^.OutF[idx] := HalfToFloat(FloatToHalf(fc[tx, 0]));
        A^.OutF[idx + 1] := HalfToFloat(FloatToHalf(fc[tx, 1]));
      end;
    end;
  end;
end;

procedure launch_fused_attention(Q, K, V: PHalf; OutputF: PSingle; SeqLen, DimHead: LongInt);
var A: TFaArgs;
begin
  A.Q := Q; A.K := K; A.V := V; A.OutF := OutputF; A.Seq := SeqLen; A.Dim := DimHead;
  Launch1D('fused_attention.gemm_online_softmax_kernel', SeqLen div FA_TILE_M, 1, FusedAttentionSmemBytes, @FaBody, @A);
end;

{ ===================== gemm_online_softmax.cu ===================== }
type
  TGoArgs = record Q, K, V: PHalf; O: PSingle; Seq, HD: LongInt; Scale: Single; end;
  PGoArgs = ^TGoArgs;

function GemmOnlineSoftmaxSmemBytes(HeadDim: LongInt): LongWord;
begin
  Result := (GO_TILE_M * (GO_TILE_K + GO_Q_PAD) + GO_TILE_N * (GO_TILE_K + GO_K_PAD) + GO_TILE_N * HeadDim) * SizeOf(THalf);
end;

function FMaxF(A, B: Single): Single; inline;
begin
  if IsNan(A) then Exit(B); if IsNan(B) then Exit(A);
  if A > B then Result := A else Result := B;
end;

procedure GoBody(const T, B, BD, GD: TDim3D; Data: Pointer);
const RQ = GO_TILE_K + GO_Q_PAD; RK = GO_TILE_K + GO_K_PAD; RPW = GO_TILE_M div GO_WARPS;
var A: PGoArgs; qBase, numK, kt, kBase, i, row, col, gr, tid, warp, lane, r, n, k, d, ws: LongInt;
    Qs, Ks, Vs: array of THalf;
    rowMax, rowDen: array[0..GO_THREADS - 1, 0..31] of Single;
    OAcc: array[0..GO_THREADS - 1, 0..31, 0..1] of Single;
    lmax, lsum: array[0..31] of Single; tmp: array[0..31] of Single; off: Integer;
    score, prevMax, pSum, p, mPrev, rm, rden: Single;
  function RowMaxAt(Th, Idx: LongInt): Single; inline;
  begin if (Idx >= 0) and (Idx < 32) then Result := rowMax[Th, Idx] else Result := NaN; end;   { (c) }
  function RowDenAt(Th, Idx: LongInt): Single; inline;
  begin if (Idx >= 0) and (Idx < 32) then Result := rowDen[Th, Idx] else Result := NaN; end;
begin
  A := PGoArgs(Data);
  qBase := B.X * GO_TILE_M;
  if qBase >= A^.Seq then Exit;
  Qs := nil; Ks := nil; Vs := nil;
  SetLength(Qs, GO_TILE_M * RQ); SetLength(Ks, GO_TILE_N * RK); SetLength(Vs, GO_TILE_N * A^.HD);   { (a) }
  for tid := 0 to GO_THREADS - 1 do
    for i := 0 to 31 do begin rowMax[tid, i] := NegInfinity; rowDen[tid, i] := 0.0; OAcc[tid, i, 0] := 0.0; OAcc[tid, i, 1] := 0.0; end; { (d) }
  numK := (A^.Seq + GO_TILE_N - 1) div GO_TILE_N;
  for kt := 0 to numK - 1 do
  begin
    kBase := kt * GO_TILE_N;
    for i := 0 to GO_TILE_N * RK - 1 do   { :166-176 (all threads jointly cover every i) }
    begin
      row := i div RK; col := i mod RK; gr := kBase + row;
      if (row < GO_TILE_N) and (col < GO_TILE_K) and (gr < A^.Seq) and (col < A^.HD) then Ks[row * RK + col] := A^.K[SizeInt(gr) * A^.HD + col]
      else Ks[row * RK + col] := FloatToHalf(0.0);
    end;
    for i := 0 to GO_TILE_M * RQ - 1 do   { :180-189 }
    begin
      row := i div RQ; col := i mod RQ; gr := qBase + row;
      if (row < GO_TILE_M) and (col < GO_TILE_K) and (gr < A^.Seq) and (col < A^.HD) then Qs[row * RQ + col] := A^.Q[SizeInt(gr) * A^.HD + col]
      else Qs[row * RQ + col] := FloatToHalf(0.0);
    end;
    for warp := 0 to GO_WARPS - 1 do   { :194-224 }
    begin
      ws := warp * RPW;
      r := 0;
      while r < RPW do
      begin
        gr := ws + r;
        if gr < GO_TILE_M then
        begin
          for lane := 0 to 31 do
          begin
            lmax[lane] := NegInfinity; lsum[lane] := 0.0;
            n := lane;
            while n < GO_TILE_N do
            begin
              score := 0.0;
              for k := 0 to GO_TILE_K - 1 do score := score + HalfToFloat(Qs[gr * RQ + k]) * HalfToFloat(Ks[n * RK + k]);
              score := score * A^.Scale;
              prevMax := lmax[lane];
              lmax[lane] := FMaxF(lmax[lane], score);
              lsum[lane] := lsum[lane] * Single(Exp(prevMax - lmax[lane])) + Single(Exp(score - lmax[lane]));
              Inc(n, 32);
            end;
          end;
          off := 16; while off > 0 do begin for lane := 0 to 31 do tmp[lane] := lmax[lane xor off]; for lane := 0 to 31 do lmax[lane] := FMaxF(lmax[lane], tmp[lane]); off := off shr 1; end;
          off := 16; while off > 0 do begin for lane := 0 to 31 do tmp[lane] := lsum[lane xor off]; for lane := 0 to 31 do lsum[lane] := lsum[lane] + tmp[lane]; off := off shr 1; end;
          for lane := 0 to 31 do   { online_softmax_update(row_max[r], row_denom[r], ...) :103-116 }
          begin
            tid := warp * 32 + lane;
            mPrev := rowMax[tid, r];
            rowMax[tid, r] := FMaxF(rowMax[tid, r], lmax[lane]);
            rowDen[tid, r] := rowDen[tid, r] * Single(Exp(mPrev - rowMax[tid, r])) + lsum[lane] * Single(Exp(lmax[lane] - rowMax[tid, r]));
          end;
        end;
        Inc(r, 16);
      end;
    end;
    for i := 0 to GO_TILE_N * A^.HD - 1 do   { :229-237 }
    begin
      row := i div A^.HD; col := i mod A^.HD; gr := kBase + row;
      if (gr < A^.Seq) and (col < A^.HD) then Vs[row * A^.HD + col] := A^.V[SizeInt(gr) * A^.HD + col];
    end;
    for warp := 0 to GO_WARPS - 1 do   { :241-252 }
    begin
      ws := warp * RPW;
      r := ws;
      while (r < ws + RPW) and (r < GO_TILE_M) do
      begin
        for lane := 0 to 31 do
        begin
          tid := warp * 32 + lane;
          d := lane;
          while d < A^.HD do
          begin
            pSum := 0.0;
            rm := RowMaxAt(tid, r);
            for n := 0 to GO_TILE_N - 1 do
            begin
              score := HalfToFloat(Qs[r * RQ + 0]) * HalfToFloat(Ks[n * RK + 0]);
              p := Single(Exp(score * A^.Scale - rm));
              pSum := pSum + p * HalfToFloat(Vs[n * A^.HD + d]);
            end;
            rden := RowDenAt(tid, r);
            OAcc[tid, r mod RPW, d mod 2] := OAcc[tid, r mod RPW, d mod 2] + pSum / rden;
            Inc(d, 32);
          end;
        end;
        Inc(r);
      end;
    end;
  end;
  for warp := 0 to GO_WARPS - 1 do   { :258-264 }
    for lane := 0 to 31 do
    begin
      tid := warp * 32 + lane;
      for r := 0 to RPW - 1 do
      begin
        gr := qBase + warp * RPW + r;
        if gr >= A^.Seq then Continue;
        d := lane;
        while d < A^.HD do begin A^.O[SizeInt(gr) * A^.HD + d] := OAcc[tid, r, d mod 2] / rowDen[tid, r]; Inc(d, 32); end;
      end;
    end;
end;

procedure launch_gemm_online_softmax(Q, K, V: PHalf; O: PSingle; SeqLen, HeadDim: LongInt);
var A: TGoArgs;
begin
  A.Q := Q; A.K := K; A.V := V; A.O := O; A.Seq := SeqLen; A.HD := HeadDim;
  A.Scale := 1.0 / Single(Sqrt(Single(HeadDim)));
  Launch1D('gemm_online_softmax_kernel', (SeqLen + GO_TILE_M - 1) div GO_TILE_M, 1, GemmOnlineSoftmaxSmemBytes(HeadDim), @GoBody, @A);
end;

end.
