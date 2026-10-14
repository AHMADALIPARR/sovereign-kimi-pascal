{ SPDX-License-Identifier: AGPL-3.0-only
  Port of the device kernels of kernels/k3_nano_harness.cu onto pascal-stack (see SK_Launch).
  Each kernel keeps the CUDA name, argument list and launch geometry; file:line refer to
  kernels/k3_nano_harness.cu. }
{$mode objfpc}{$H+}{$Q-}{$R-}{$POINTERMATH ON}
unit SK_K3Kernels;

interface

uses SysUtils, Math, PascalGPU_Types, SK_Half, SK_Launch;

const
  WARP_SIZE = 32; MAX_TOP_K = 8; MAX_LAYERS = 32; MAX_EXPERTS = 128; MAX_DIM = 8192;
  MAX_VOCAB = 65536; MAX_SEQ_LEN = 4096; MAX_GAMMA = 8; WMMA_M = 16; WMMA_N = 16; WMMA_K = 16; { :38-48 }

function DivCeil(A, B: LongInt): LongInt; inline;   { :118 }

{ Launch wrappers: same parameters as the CUDA kernels plus the <<<grid, block>>> geometry. }
procedure rmsnorm_kernel(Grid, Block: LongWord; Input, Weight, Output: PSingle; Eps: Single; Dim: LongInt);       { :142-164 }
procedure qkv_proj_kernel(Grid, Block: LongWord; Input, Weight, QkvOut: PSingle; Dim: LongInt);                  { :170-183 }
procedure kimi_delta_attention_kernel(Grid, Block: LongWord; Q, PrevK, CurrK, V, OutP: PSingle; Scale: Single; Dim: LongInt); { :189-202 }
procedure attn_out_proj_kernel(Grid, Block: LongWord; Input, Weight, Output: PSingle; Dim: LongInt);            { :208-221 }
procedure latent_moe_router_kernel(Grid, Block: LongWord; Input, RouterWeights: PSingle; SelectedExperts: PLongInt;
  RoutingWeights: PSingle; Dim, NumExperts, TopK: LongInt);                                                       { :227-261 }
procedure warp_decode_moe_kernel(Grid, Block: LongWord; Input, ExpertW1, ExpertW2: PSingle; SelectedExperts: PLongInt;
  RoutingWeights, Output: PSingle; Dim, IntermediateDim, TopK: LongInt);                                          { :267-295 }
procedure wmma_decode_moe_kernel(Grid, Block: LongWord; Input, ExpertW1, ExpertW2: PHalf; SelectedExperts: PLongInt;
  RoutingWeights, Output: PSingle; Dim, IntermediateDim, TopK, DimPadded, InterPadded: LongInt);                 { :301-385 }
procedure convert_fp32_to_fp16_kernel(Grid, Block: LongWord; Input: PSingle; Output: PHalf; Dim, DimPadded: LongInt); { :391-414 }
procedure output_head_kernel(Grid, Block: LongWord; Input, Weight, Logits: PSingle; VocabSize, Dim: LongInt);   { :420-433 }
procedure top_p_sample_kernel(Grid, Block: LongWord; Logits: PSingle; VocabSize: LongInt; Temperature, TopP: Single;
  Seed: QWord; SampledToken: PLongInt);                                                                            { :439-476 }
procedure speculative_acceptance_kernel(Grid, Block: LongWord; TargetLogits, DraftLogits: PSingle; DraftTokens: PLongInt;
  VocabSize, Gamma: LongInt; Temperature: Single; Seed: QWord; AcceptCount, AcceptedTokens: PLongInt);         { :482-533 }
procedure batched_embedding_kernel(Grid, Block: LongWord; TokenIds: PLongInt; TokenEmb, PosEmb, Residual: PSingle;
  Dim, MaxSeqLen, StartPos, Batch: LongInt);                                                                        { :539-555 }
procedure embedding_kernel(Grid, Block: LongWord; TokenId, Pos: LongInt; TokenEmb, PosEmb, Residual: PSingle;
  Dim, MaxSeqLen: LongInt);                                                                                         { :561-570 }

{ __shfl_xor_sync butterfly over a 32-lane register array (:124-136) }
procedure WarpReduceSum(var V: array of Single);
procedure WarpReduceMax(var V: array of Single);
function XorShiftRng(var S: QWord): Single;   { :466-468 / :492-497 }
function SiLU(X: Single): Single; inline;

implementation

function DivCeil(A, B: LongInt): LongInt; inline;
begin Result := (A + B - 1) div B; end;

function SiLU(X: Single): Single; inline;
begin Result := X / (1.0 + Single(Exp(-X))); end;

procedure WarpReduceSum(var V: array of Single);
var off, l: Integer; tmp: array[0..31] of Single;
begin
  Assert(Length(V) = 32);
  off := 16;
  while off > 0 do
  begin
    for l := 0 to 31 do tmp[l] := V[l xor off];
    for l := 0 to 31 do V[l] := V[l] + tmp[l];
    off := off div 2;
  end;
end;

function FMaxF(A, B: Single): Single; inline;
begin
  { CUDA fmaxf: if one operand is NaN, return the other }
  if IsNan(A) then Exit(B);
  if IsNan(B) then Exit(A);
  if A > B then Result := A else Result := B;
end;

function FMinF(A, B: Single): Single; inline;
begin
  if IsNan(A) then Exit(B);
  if IsNan(B) then Exit(A);
  if A < B then Result := A else Result := B;
end;

procedure WarpReduceMax(var V: array of Single);
var off, l: Integer; tmp: array[0..31] of Single;
begin
  off := 16;
  while off > 0 do
  begin
    for l := 0 to 31 do tmp[l] := V[l xor off];
    for l := 0 to 31 do V[l] := FMaxF(V[l], tmp[l]);
    off := off div 2;
  end;
end;

function XorShiftRng(var S: QWord): Single;
var x: QWord;
begin
  x := S;
  x := x xor (x shr 12); x := x xor (x shl 25); x := x xor (x shr 27);
  S := x;
  Result := Single(LongWord((x * QWord($2545F4914F6CDD1D)) shr 32)) / Single(4294967296.0);
end;

{ ---------------- rmsnorm (block-cooperative: one pascal-stack thread per CUDA block) ---------------- }
type
  TRmsArgs = record Input, Weight, Output: PSingle; Eps: Single; Dim: LongInt; CudaBlock: LongInt; end;
  PRmsArgs = ^TRmsArgs;

procedure RmsBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PRmsArgs; s: array of Single; tid, i, stride: LongInt; x, rms: Single;
begin
  A := PRmsArgs(Data);
  s := nil; SetLength(s, A^.CudaBlock);
  for tid := 0 to A^.CudaBlock - 1 do
  begin
    s[tid] := 0.0; i := tid;
    while i < A^.Dim do begin x := A^.Input[i]; s[tid] := s[tid] + x * x; Inc(i, A^.CudaBlock); end;
  end;
  { __syncthreads }
  stride := A^.CudaBlock div 2;
  while stride > 0 do
  begin
    for tid := 0 to stride - 1 do s[tid] := s[tid] + s[tid + stride];
    stride := stride shr 1;
  end;
  rms := 1.0 / Single(Sqrt(s[0] / Single(A^.Dim) + A^.Eps));   { rsqrtf }
  for tid := 0 to A^.CudaBlock - 1 do
  begin
    i := tid;
    while i < A^.Dim do begin A^.Output[i] := A^.Input[i] * rms * A^.Weight[i]; Inc(i, A^.CudaBlock); end;
  end;
end;

procedure rmsnorm_kernel(Grid, Block: LongWord; Input, Weight, Output: PSingle; Eps: Single; Dim: LongInt);
var A: TRmsArgs;
begin
  A.Input := Input; A.Weight := Weight; A.Output := Output; A.Eps := Eps; A.Dim := Dim; A.CudaBlock := Block;
  Launch1D('rmsnorm_kernel', Grid, 1, Block * SizeOf(Single), @RmsBody, @A);
end;

{ ---------------- row-dot kernels (thread granularity) ---------------- }
type
  TDotArgs = record Input, Weight, Output: PSingle; Rows, Dim: LongInt; end;
  PDotArgs = ^TDotArgs;

procedure DotBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PDotArgs; row, i: LongInt; sum: Single; w: PSingle;
begin
  A := PDotArgs(Data);
  row := LongInt(B.X * BD.X + T.X);
  if row >= A^.Rows then Exit;
  sum := 0.0;
  w := A^.Weight + PtrInt(row) * A^.Dim;
  for i := 0 to A^.Dim - 1 do sum := sum + w[i] * A^.Input[i];
  A^.Output[row] := sum;
end;

procedure LaunchDot(const N: AnsiString; Grid, Block: LongWord; Input, Weight, Output: PSingle; Rows, Dim: LongInt);
var A: TDotArgs;
begin
  { pascal-stack alias guard: the CUDA source passes the same buffer as input and output for
    attn_out_proj (:870); with sequential threads that would read already-overwritten inputs. }
  Assert((Input <> Output), N + ': input and output must be disjoint (snapshot the input first)');
  A.Input := Input; A.Weight := Weight; A.Output := Output; A.Rows := Rows; A.Dim := Dim;
  Launch1D(N, Grid, Block, 0, @DotBody, @A);
end;

procedure qkv_proj_kernel(Grid, Block: LongWord; Input, Weight, QkvOut: PSingle; Dim: LongInt);
begin LaunchDot('qkv_proj_kernel', Grid, Block, Input, Weight, QkvOut, 3 * Dim, Dim); end;

procedure attn_out_proj_kernel(Grid, Block: LongWord; Input, Weight, Output: PSingle; Dim: LongInt);
begin LaunchDot('attn_out_proj_kernel', Grid, Block, Input, Weight, Output, Dim, Dim); end;

procedure output_head_kernel(Grid, Block: LongWord; Input, Weight, Logits: PSingle; VocabSize, Dim: LongInt);
begin LaunchDot('output_head_kernel', Grid, Block, Input, Weight, Logits, VocabSize, Dim); end;

{ ---------------- delta attention ---------------- }
type
  TDeltaArgs = record Q, PrevK, CurrK, V, OutP: PSingle; Scale: Single; Dim: LongInt; end;
  PDeltaArgs = ^TDeltaArgs;

procedure DeltaBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PDeltaArgs; tid: LongInt; dk, w: Single;
begin
  A := PDeltaArgs(Data);
  tid := LongInt(B.X * BD.X + T.X);
  if tid >= A^.Dim then Exit;
  dk := A^.CurrK[tid] - A^.PrevK[tid];
  w := Single(Tanh(A^.Q[tid] * dk * A^.Scale));
  A^.OutP[tid] := w * A^.V[tid];
end;

procedure kimi_delta_attention_kernel(Grid, Block: LongWord; Q, PrevK, CurrK, V, OutP: PSingle; Scale: Single; Dim: LongInt);
var A: TDeltaArgs;
begin
  A.Q := Q; A.PrevK := PrevK; A.CurrK := CurrK; A.V := V; A.OutP := OutP; A.Scale := Scale; A.Dim := Dim;
  Launch1D('kimi_delta_attention_kernel', Grid, Block, 0, @DeltaBody, @A);
end;

{ ---------------- latent MoE router ---------------- }
type
  TRouterArgs = record Input, RW: PSingle; Sel: PLongInt; RWt: PSingle; Dim, NE, TopK: LongInt; end;
  PRouterArgs = ^TRouterArgs;

procedure RouterBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PRouterArgs; logits: array of Single; e, d, k, maxIdx: LongInt; score, maxVal, sumExp: Single; w: PSingle;
begin
  if T.X <> 0 then Exit;
  A := PRouterArgs(Data);
  logits := nil; SetLength(logits, A^.NE);
  for e := 0 to A^.NE - 1 do
  begin
    score := 0.0; w := A^.RW + PtrInt(e) * A^.Dim;
    for d := 0 to A^.Dim - 1 do score := score + A^.Input[d] * w[d];
    logits[e] := score;
  end;
  for k := 0 to A^.TopK - 1 do
  begin
    maxVal := -1e9; maxIdx := -1;
    for e := 0 to A^.NE - 1 do
      if logits[e] > maxVal then begin maxVal := logits[e]; maxIdx := e; end;
    A^.Sel[k] := maxIdx; A^.RWt[k] := maxVal;
    if maxIdx <> -1 then logits[maxIdx] := -1e9;
  end;
  sumExp := 0.0;
  for k := 0 to A^.TopK - 1 do begin A^.RWt[k] := Single(Exp(A^.RWt[k])); sumExp := sumExp + A^.RWt[k]; end;
  for k := 0 to A^.TopK - 1 do A^.RWt[k] := A^.RWt[k] / sumExp;
end;

procedure latent_moe_router_kernel(Grid, Block: LongWord; Input, RouterWeights: PSingle; SelectedExperts: PLongInt;
  RoutingWeights: PSingle; Dim, NumExperts, TopK: LongInt);
var A: TRouterArgs;
begin
  A.Input := Input; A.RW := RouterWeights; A.Sel := SelectedExperts; A.RWt := RoutingWeights;
  A.Dim := Dim; A.NE := NumExperts; A.TopK := TopK;
  Launch1D('latent_moe_router_kernel', Grid, Block, NumExperts * SizeOf(Single), @RouterBody, @A);
end;

{ ---------------- warp-decode MoE (warp granularity) ---------------- }
type
  TWarpMoeArgs = record Input, W1, W2: PSingle; Sel: PLongInt; RWt, OutP: PSingle; Dim, Inter, TopK, CudaBlock: LongInt; end;
  PWarpMoeArgs = ^TWarpMoeArgs;

procedure WarpMoeBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PWarpMoeArgs; warpId, lane, k, j, i, ex: LongInt; acc: array[0..31] of Single; rw, h: Single; w1, w2: PSingle;
begin
  A := PWarpMoeArgs(Data);
  { pascal-stack thread T.X is CUDA warp T.X of block B.X }
  warpId := LongInt((B.X * LongWord(A^.CudaBlock) + T.X * 32) div 32);
  if warpId >= A^.Dim then Exit;
  for lane := 0 to 31 do
  begin
    acc[lane] := 0.0;
    for k := 0 to A^.TopK - 1 do
    begin
      ex := A^.Sel[k]; rw := A^.RWt[k];
      w1 := A^.W1 + PtrInt(ex) * A^.Dim * A^.Inter;
      w2 := A^.W2 + PtrInt(ex) * A^.Inter * A^.Dim;
      j := lane;
      while j < A^.Inter do
      begin
        h := 0.0;
        for i := 0 to A^.Dim - 1 do h := h + A^.Input[i] * w1[PtrInt(i) * A^.Inter + j];
        h := h / (1.0 + Single(Exp(-h)));
        acc[lane] := acc[lane] + h * w2[PtrInt(j) * A^.Dim + warpId] * rw;
        Inc(j, WARP_SIZE);
      end;
    end;
  end;
  WarpReduceSum(acc);
  A^.OutP[warpId] := acc[0];
end;

procedure warp_decode_moe_kernel(Grid, Block: LongWord; Input, ExpertW1, ExpertW2: PSingle; SelectedExperts: PLongInt;
  RoutingWeights, Output: PSingle; Dim, IntermediateDim, TopK: LongInt);
var A: TWarpMoeArgs;
begin
  Assert(Block mod 32 = 0, 'warp_decode_moe_kernel requires whole warps');
  A.Input := Input; A.W1 := ExpertW1; A.W2 := ExpertW2; A.Sel := SelectedExperts; A.RWt := RoutingWeights;
  A.OutP := Output; A.Dim := Dim; A.Inter := IntermediateDim; A.TopK := TopK; A.CudaBlock := Block;
  Launch1D('warp_decode_moe_kernel', Grid, Block div 32, 0, @WarpMoeBody, @A);
end;

{ ---------------- WMMA FP16 MoE (block granularity, warps simulated serially) ----------------
  Fragment semantics: wmma 16x16x16, A row_major, B col_major (B[k][n] = ptr[n*ld + k]),
  FP32 accumulate; products of two halves are exact in FP32, accumulation is k-sequential.
  Shared memory is zero-initialised per block (uninitialised in CUDA). }
type
  TWmmaArgs = record Input, W1, W2: PHalf; Sel: PLongInt; RWt, OutP: PSingle;
    Dim, Inter, TopK, DimP, InterP, CudaBlock: LongInt; end;
  PWmmaArgs = ^TWmmaArgs;
  TFrag = array[0..15, 0..15] of Single;

procedure MmaRowCol(var C: TFrag; const ARowMajor: PHalf; LdA: LongInt; const BColMajor: PHalf; LdB: LongInt);
var m, n, k: Integer; s: Single;
begin
  for m := 0 to 15 do
    for n := 0 to 15 do
    begin
      s := C[m, n];
      for k := 0 to 15 do
        s := s + HalfToFloat(ARowMajor[m * LdA + k]) * HalfToFloat(BColMajor[PtrInt(n) * LdB + k]);
      C[m, n] := s;
    end;
end;

procedure WmmaBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PWmmaArgs; nWarps, w, warpId, lane, outTiles, outStart, validOut, k, ex, jt, it, jStart, iStart, i, row, col: LongInt;
    sInput, sHidden: array of THalf; sInBc, sHidBc: array[0..255] of THalf; sHidFrag, sOutFrag: array[0..255] of Single;
    outFrag, hidFrag: TFrag; active: array of Boolean; rw, val, silu: Single; w1, w2: PHalf;
begin
  A := PWmmaArgs(Data);
  nWarps := A^.CudaBlock div 32;
  outTiles := (A^.DimP + WMMA_M - 1) div WMMA_M;
  sInput := nil; sHidden := nil; active := nil;
  SetLength(sInput, A^.DimP); SetLength(sHidden, A^.InterP); SetLength(active, nWarps);
  FillChar(sInBc, SizeOf(sInBc), 0); FillChar(sHidBc, SizeOf(sHidBc), 0);
  FillChar(sHidFrag, SizeOf(sHidFrag), 0); FillChar(sOutFrag, SizeOf(sOutFrag), 0);
  { phase 1 (:311-324): warps beyond output_tiles return before loading s_input }
  for w := 0 to nWarps - 1 do
  begin
    warpId := LongInt(B.X) * nWarps + w;
    active[w] := warpId < outTiles;
    if not active[w] then Continue;
    for lane := 0 to 31 do
    begin
      i := w * 32 + lane;
      while i < A^.DimP do begin sInput[i] := A^.Input[i]; Inc(i, A^.CudaBlock); end;
    end;
  end;
  { phase 2: each active warp, serially }
  for w := 0 to nWarps - 1 do
  begin
    if not active[w] then Continue;
    warpId := LongInt(B.X) * nWarps + w;
    outStart := warpId * WMMA_M;
    validOut := Min(WMMA_M, A^.Dim - outStart);
    FillChar(outFrag, SizeOf(outFrag), 0);
    for k := 0 to A^.TopK - 1 do
    begin
      ex := A^.Sel[k]; rw := A^.RWt[k];
      w1 := A^.W1 + PtrInt(ex) * A^.DimP * A^.InterP;
      w2 := A^.W2 + PtrInt(ex) * A^.InterP * A^.DimP;
      for jt := 0 to (A^.InterP + WMMA_N - 1) div WMMA_N - 1 do
      begin
        jStart := jt * WMMA_N;
        FillChar(hidFrag, SizeOf(hidFrag), 0);
        for it := 0 to (A^.DimP + WMMA_K - 1) div WMMA_K - 1 do
        begin
          iStart := it * WMMA_K;
          for lane := 0 to 31 do   { lane_id < 256 always true; lanes fill rows 0..1 }
          begin
            row := lane div WMMA_K; col := lane mod WMMA_K;
            if iStart + col < A^.DimP then sInBc[row * WMMA_K + col] := sInput[iStart + col]
            else sInBc[row * WMMA_K + col] := FloatToHalf(0.0);
          end;
          MmaRowCol(hidFrag, @sInBc[0], WMMA_K, w1 + PtrInt(iStart) * A^.InterP + jStart, A^.InterP);
        end;
        for row := 0 to 15 do for col := 0 to 15 do sHidFrag[row * WMMA_N + col] := hidFrag[row, col];
        for lane := 0 to WMMA_N - 1 do
          if jStart + lane < A^.Inter then
          begin
            val := sHidFrag[lane];
            silu := val / (1.0 + Single(Exp(-val)));
            sHidden[jStart + lane] := FloatToHalf(silu * rw);
          end;
      end;
      for jt := 0 to (A^.InterP + WMMA_K - 1) div WMMA_K - 1 do
      begin
        jStart := jt * WMMA_K;
        for lane := 0 to 31 do
        begin
          row := lane div WMMA_K; col := lane mod WMMA_K;
          if jStart + col < A^.InterP then sHidBc[row * WMMA_K + col] := sHidden[jStart + col]
          else sHidBc[row * WMMA_K + col] := FloatToHalf(0.0);
        end;
        MmaRowCol(outFrag, @sHidBc[0], WMMA_K, w2 + PtrInt(jStart) * A^.DimP + outStart, A^.DimP);
      end;
    end;
    for row := 0 to 15 do for col := 0 to 15 do sOutFrag[row * WMMA_N + col] := outFrag[row, col];
    for lane := 0 to 31 do
      if lane < validOut then A^.OutP[outStart + lane] := sOutFrag[lane];
  end;
end;

procedure wmma_decode_moe_kernel(Grid, Block: LongWord; Input, ExpertW1, ExpertW2: PHalf; SelectedExperts: PLongInt;
  RoutingWeights, Output: PSingle; Dim, IntermediateDim, TopK, DimPadded, InterPadded: LongInt);
var A: TWmmaArgs;
begin
  Assert(Block mod 32 = 0);
  A.Input := Input; A.W1 := ExpertW1; A.W2 := ExpertW2; A.Sel := SelectedExperts; A.RWt := RoutingWeights;
  A.OutP := Output; A.Dim := Dim; A.Inter := IntermediateDim; A.TopK := TopK; A.DimP := DimPadded;
  A.InterP := InterPadded; A.CudaBlock := Block;
  Launch1D('wmma_decode_moe_kernel', Grid, 1, (DimPadded + InterPadded) * SizeOf(THalf), @WmmaBody, @A);
end;

{ ---------------- fp32 -> fp16 ---------------- }
type
  TConvArgs = record Input: PSingle; OutP: PHalf; Dim, DimP: LongInt; end;
  PConvArgs = ^TConvArgs;

procedure ConvBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PConvArgs; idx, stride, i, tail: LongInt; f: array[0..3] of Single; j: Integer;
begin
  A := PConvArgs(Data);
  idx := LongInt(B.X * BD.X + T.X);
  stride := LongInt(GD.X * BD.X);
  i := idx * 4;
  while i < A^.Dim do
  begin
    for j := 0 to 3 do if i + j < A^.Dim then f[j] := A^.Input[i + j] else f[j] := 0.0;
    for j := 0 to 3 do if i + j < A^.DimP then A^.OutP[i + j] := FloatToHalf(f[j]);
    Inc(i, stride * 4);
  end;
  tail := ((A^.Dim + 3) div 4) * 4;
  i := tail + idx;
  while i < A^.DimP do begin A^.OutP[i] := FloatToHalf(0.0); Inc(i, stride); end;
end;

procedure convert_fp32_to_fp16_kernel(Grid, Block: LongWord; Input: PSingle; Output: PHalf; Dim, DimPadded: LongInt);
var A: TConvArgs;
begin
  A.Input := Input; A.OutP := Output; A.Dim := Dim; A.DimP := DimPadded;
  Launch1D('convert_fp32_to_fp16_kernel', Grid, Block, 0, @ConvBody, @A);
end;

{ ---------------- top-p sampling ---------------- }
type
  TTopPArgs = record Logits: PSingle; Vocab: LongInt; Temp, TopP: Single; Seed: QWord; Tok: PLongInt; end;
  PTopPArgs = ^TTopPArgs;

procedure TopPBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PTopPArgs; i, cutoff, token: LongInt; maxL, sumExp, cum, topMass, r: Single; x: QWord;
begin
  if (T.X <> 0) or (B.X <> 0) then Exit;
  A := PTopPArgs(Data);
  maxL := -1e9;
  for i := 0 to A^.Vocab - 1 do maxL := FMaxF(maxL, A^.Logits[i]);
  sumExp := 0.0;
  for i := 0 to A^.Vocab - 1 do begin A^.Logits[i] := Single(Exp((A^.Logits[i] - maxL) / A^.Temp)); sumExp := sumExp + A^.Logits[i]; end;
  for i := 0 to A^.Vocab - 1 do A^.Logits[i] := A^.Logits[i] / sumExp;
  cum := 0.0; cutoff := A^.Vocab - 1;
  for i := 0 to A^.Vocab - 1 do begin cum := cum + A^.Logits[i]; if cum >= A^.TopP then begin cutoff := i; Break; end; end;
  topMass := 0.0;
  for i := 0 to cutoff do topMass := topMass + A^.Logits[i];
  for i := 0 to cutoff do A^.Logits[i] := A^.Logits[i] / topMass;
  for i := cutoff + 1 to A^.Vocab - 1 do A^.Logits[i] := 0.0;
  x := A^.Seed;
  r := XorShiftRng(x);
  cum := 0.0; token := 0;
  for i := 0 to cutoff do begin cum := cum + A^.Logits[i]; if r < cum then begin token := i; Break; end; end;
  A^.Tok^ := token;
end;

procedure top_p_sample_kernel(Grid, Block: LongWord; Logits: PSingle; VocabSize: LongInt; Temperature, TopP: Single;
  Seed: QWord; SampledToken: PLongInt);
var A: TTopPArgs;
begin
  A.Logits := Logits; A.Vocab := VocabSize; A.Temp := Temperature; A.TopP := TopP; A.Seed := Seed; A.Tok := SampledToken;
  Launch1D('top_p_sample_kernel', Grid, Block, 0, @TopPBody, @A);
end;

{ ---------------- speculative acceptance ---------------- }
type
  TSpecArgs = record Tgt, Drf: PSingle; DTok: PLongInt; Vocab, Gamma: LongInt; Temp: Single; Seed: QWord; Cnt, Acc: PLongInt; end;
  PSpecArgs = ^TSpecArgs;

procedure SpecBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PSpecArgs; st: QWord; accepted, i, v, token, sampled: LongInt; tgt, drft: PSingle;
    maxT, maxD, sumT, sumD, pT, pD, ap, u, r, cum: Single;
begin
  if (T.X <> 0) or (B.X <> 0) then Exit;
  A := PSpecArgs(Data);
  st := A^.Seed; accepted := 0;
  for i := 0 to A^.Gamma - 1 do
  begin
    token := A^.DTok[i];
    tgt := A^.Tgt + PtrInt(i + 1) * A^.Vocab;
    drft := A^.Drf + PtrInt(i) * A^.Vocab;
    maxT := -1e9; maxD := -1e9;
    for v := 0 to A^.Vocab - 1 do begin maxT := FMaxF(maxT, tgt[v]); maxD := FMaxF(maxD, drft[v]); end;
    sumT := 0; sumD := 0;
    for v := 0 to A^.Vocab - 1 do
    begin
      sumT := sumT + Single(Exp((tgt[v] - maxT) / A^.Temp));
      sumD := sumD + Single(Exp((drft[v] - maxD) / A^.Temp));
    end;
    pT := Single(Exp((tgt[token] - maxT) / A^.Temp)) / sumT;
    pD := Single(Exp((drft[token] - maxD) / A^.Temp)) / sumD;
    ap := FMinF(1.0, pT / (pD + Single(1e-10)));
    u := XorShiftRng(st);
    if u < ap then begin A^.Acc[accepted] := token; Inc(accepted); end
    else
    begin
      r := XorShiftRng(st); cum := 0; sampled := 0;
      for v := 0 to A^.Vocab - 1 do
      begin
        cum := cum + Single(Exp((tgt[v] - maxT) / A^.Temp)) / sumT;
        if r < cum then begin sampled := v; Break; end;
      end;
      A^.Acc[accepted] := sampled; Inc(accepted);
      Break;
    end;
  end;
  A^.Cnt^ := accepted;
end;

procedure speculative_acceptance_kernel(Grid, Block: LongWord; TargetLogits, DraftLogits: PSingle; DraftTokens: PLongInt;
  VocabSize, Gamma: LongInt; Temperature: Single; Seed: QWord; AcceptCount, AcceptedTokens: PLongInt);
var A: TSpecArgs;
begin
  A.Tgt := TargetLogits; A.Drf := DraftLogits; A.DTok := DraftTokens; A.Vocab := VocabSize; A.Gamma := Gamma;
  A.Temp := Temperature; A.Seed := Seed; A.Cnt := AcceptCount; A.Acc := AcceptedTokens;
  Launch1D('speculative_acceptance_kernel', Grid, Block, 0, @SpecBody, @A);
end;

{ ---------------- embeddings ---------------- }
type
  TEmbArgs = record Ids: PLongInt; TokId, Pos: LongInt; TE, PE, Res: PSingle; Dim, MaxSeq, Start, Batch: LongInt; end;
  PEmbArgs = ^TEmbArgs;

procedure BatchedEmbBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PEmbArgs; p, tok, sp, i: LongInt;
begin
  A := PEmbArgs(Data);
  p := LongInt(B.X);
  if p >= A^.Batch then Exit;
  tok := A^.Ids[p]; sp := A^.Start + p;
  i := LongInt(T.X);
  while i < A^.Dim do
  begin
    A^.Res[PtrInt(p) * A^.Dim + i] := A^.TE[PtrInt(tok) * A^.Dim + i] + A^.PE[PtrInt(sp) * A^.Dim + i];
    Inc(i, LongInt(BD.X));
  end;
end;

procedure batched_embedding_kernel(Grid, Block: LongWord; TokenIds: PLongInt; TokenEmb, PosEmb, Residual: PSingle;
  Dim, MaxSeqLen, StartPos, Batch: LongInt);
var A: TEmbArgs;
begin
  A.Ids := TokenIds; A.TE := TokenEmb; A.PE := PosEmb; A.Res := Residual; A.Dim := Dim; A.MaxSeq := MaxSeqLen;
  A.Start := StartPos; A.Batch := Batch;
  Launch1D('batched_embedding_kernel', Grid, Block, 0, @BatchedEmbBody, @A);
end;

procedure EmbBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PEmbArgs; tid: LongInt;
begin
  A := PEmbArgs(Data);
  tid := LongInt(T.X);   { the CUDA kernel uses threadIdx.x only }
  if tid < A^.Dim then
    A^.Res[tid] := A^.TE[PtrInt(A^.TokId) * A^.Dim + tid] + A^.PE[PtrInt(A^.Pos) * A^.Dim + tid];
end;

procedure embedding_kernel(Grid, Block: LongWord; TokenId, Pos: LongInt; TokenEmb, PosEmb, Residual: PSingle;
  Dim, MaxSeqLen: LongInt);
var A: TEmbArgs;
begin
  A.TokId := TokenId; A.Pos := Pos; A.TE := TokenEmb; A.PE := PosEmb; A.Res := Residual; A.Dim := Dim; A.MaxSeq := MaxSeqLen;
  Launch1D('embedding_kernel', Grid, Block, 0, @EmbBody, @A);
end;

end.
