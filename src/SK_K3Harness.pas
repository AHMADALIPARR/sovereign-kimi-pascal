{ SPDX-License-Identifier: AGPL-3.0-only
  Port of the HOST side of kernels/k3_nano_harness.cu (:54-104 structs, :576-681 loader,
  :687-747 fp16 conversion, :753-784 init/free, :790-813 tokenizer, :819-890 forward_token,
  :896-904 prefill, :910-1070 draft model, :1076-1118 speculative decode, :1124-1142 FFI).
  "Device memory" is a Pascal dynamic array owned by exactly one record field (no copied
  TDeviceMemory handles); cudaMemcpy D2D/H2D become Move on those arrays. }
{$mode objfpc}{$H+}{$Q-}{$R-}{$POINTERMATH ON}
unit SK_K3Harness;

interface

uses SysUtils, Classes, Math, SK_Half, SK_K3Kernels;

type
  TSingleArray = array of Single;
  THalfArray = array of THalf;
  TIntArray = array of LongInt;

  { :54-64 — field order matters: load_draft_model freads this struct directly (:921) }
  TK3Config = packed record
    Dim, IntermediateDim, NumExperts, TopK, NumLayers, VocabSize, MaxSeqLen, TiedEmbeddings: LongInt;
    RmsNormEps: Single;
  end;

  TK3Layer = record
    AttnLn, Qkv, AttnOut, MoeLn, Router, W1, W2: TSingleArray;
    W1Fp16, W2Fp16: THalfArray;
  end;

  { :66-82 }
  TK3Weights = record
    TokenEmb, PosEmb, LnF, OutputW: TSingleArray;
    Layers: array of TK3Layer;
    DimPadded, InterPadded: LongInt;
  end;

  { :84-89 }
  TKVCache = record KCache, VCache, DeltaK: TSingleArray; SeqLen: LongInt; end;

  { :91-104 }
  TK3Model = class
    Config: TK3Config;
    Weights: TK3Weights;
    KV: TKVCache;
    Residual, AttnOut, MoeOut, QkvBuffer, LnOut, Logits: TSingleArray;
    SelectedExperts: TIntArray;
    RoutingWeights: TSingleArray;
    ResidualFp16: THalfArray;
  end;

  { :910-920 — shares token/pos embeddings with the target (pointers, not copies) }
  TDraftModel = class
    Config: TK3Config;
    Weights: TK3Weights;
    Target: TK3Model;
    KV: TKVCache;
    Residual, LnOut, QkvBuffer, AttnOut, FfnOut, Logits: TSingleArray;
  end;

  EK3Harness = class(Exception);

procedure LoadModelWeights(const Path: AnsiString; out Config: TK3Config; out W: TK3Weights; Verbose: Boolean = True); { :585-681 }
procedure ConvertAndUploadFp16Weights(M: TK3Model; Verbose: Boolean = True);   { :687-747 }
procedure InitModel(M: TK3Model; const Config: TK3Config);                      { :753-774 }
procedure FreeModel(M: TK3Model);                                               { :776-784 }
function CharToToken(C: Byte): LongInt;                                         { :796 }
function TokenToChar(Tok: LongInt): AnsiChar;                                   { :797-799 }
procedure TokenizePrompt(const Prompt: AnsiString; var Tokens: TIntArray; out NTokens: LongInt; MaxTokens: LongInt); { :801-807 }
procedure DetokenizeAndPrint(Token: LongInt);                                   { :809-813 }
procedure ForwardToken(M: TK3Model; TokenId, Pos: LongInt; LogitsOut: PSingle);  { :819-890 }
procedure Prefill(M: TK3Model; const Tokens: TIntArray; Len: LongInt);          { :896-904 }
procedure LoadDraftModel(const Path: AnsiString; D: TDraftModel; Target: TK3Model); { :922-1001 }
function DraftStep(D: TDraftModel; TokenId, Pos: LongInt): LongInt;             { :1003-1070 }
procedure SpeculativeDecode(Target: TK3Model; D: TDraftModel; PromptLen, MaxNewTokens: LongInt;
  Temperature, TopP: Single; Gamma: LongInt; Seed: QWord);                      { :1076-1118 }

{ FFI entry points (:1124-1142) with the C ABI expected by ffi.rs }
function k3_engine_init(ModelPath: PAnsiChar): Pointer; cdecl;
function k3_engine_forward(Engine: Pointer; Token, Pos: LongInt; OutLogits: PSingle): LongInt; cdecl;
procedure k3_engine_free(Engine: Pointer); cdecl;

const
  TOK_PAD = 0; TOK_BOS = 1; TOK_EOS = 2; TOK_UNK = 3; TOK_BYTE_BASE = 4;   { :790-794 }

implementation

const HEADER_SIZE = 40;   { sizeof(ModelHeader) :576-583 }

procedure ReadExact(S: TStream; var Buf; Bytes: SizeInt);
var n: SizeInt;
begin
  if Bytes <= 0 then Exit;
  n := S.Read(Buf, Bytes);
  { fread() short reads are silently ignored by the C code; here they are an error. }
  if n <> Bytes then raise EK3Harness.CreateFmt('short read: wanted %d bytes, got %d', [Bytes, n]);
end;

procedure ReadFloats(S: TStream; var A: TSingleArray; Count: SizeInt);
begin
  A := nil; SetLength(A, Count);
  if Count > 0 then ReadExact(S, A[0], Count * SizeOf(Single));
end;

procedure LoadModelWeights(const Path: AnsiString; out Config: TK3Config; out W: TK3Weights; Verbose: Boolean);
var F: TFileStream; Hdr: array[0..HEADER_SIZE - 1] of Byte; I: PLongInt; dim, vocab, inter, experts, layers, l: SizeInt;
begin
  F := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    ReadExact(F, Hdr, HEADER_SIZE);
    if (Hdr[0] <> Ord('K')) or (Hdr[1] <> Ord('3')) or (Hdr[2] <> Ord('M')) or (Hdr[3] <> Ord('1')) then
      raise EK3Harness.Create('Invalid magic');
    I := PLongInt(@Hdr[4]);   { version, dim, num_layers, num_experts, top_k, vocab, max_seq, inter }
    Config.Dim := I[1]; Config.NumLayers := I[2]; Config.NumExperts := I[3]; Config.TopK := I[4];
    Config.VocabSize := I[5]; Config.MaxSeqLen := I[6]; Config.IntermediateDim := I[7];
    Config.TiedEmbeddings := ShortInt(Hdr[36]);
    Config.RmsNormEps := 1e-6;
    if Verbose then
      WriteLn(Format('Config: dim=%d layers=%d experts=%d top_k=%d vocab=%d max_seq=%d inter=%d',
        [Config.Dim, Config.NumLayers, Config.NumExperts, Config.TopK, Config.VocabSize, Config.MaxSeqLen, Config.IntermediateDim]));
    dim := Config.Dim; vocab := Config.VocabSize; inter := Config.IntermediateDim;
    experts := Config.NumExperts; layers := Config.NumLayers;
    ReadFloats(F, W.TokenEmb, vocab * dim);
    ReadFloats(F, W.PosEmb, SizeInt(Config.MaxSeqLen) * dim);
    ReadFloats(F, W.LnF, dim);
    if Config.TiedEmbeddings = 0 then ReadFloats(F, W.OutputW, vocab * dim)
    else begin W.OutputW := nil; SetLength(W.OutputW, vocab * dim); end;
    { NB (:613,:617,:626): with tied embeddings the C code uploads the UNINITIALISED malloc buffer
      h_output_w as output_weight; it never aliases token_emb. Here it is zero-filled. }
    W.Layers := nil; SetLength(W.Layers, layers);
    for l := 0 to layers - 1 do
      with W.Layers[l] do
      begin
        ReadFloats(F, AttnLn, dim); ReadFloats(F, Qkv, 3 * dim * dim); ReadFloats(F, AttnOut, dim * dim);
        ReadFloats(F, MoeLn, dim); ReadFloats(F, Router, experts * dim);
        ReadFloats(F, W1, experts * dim * inter); ReadFloats(F, W2, experts * inter * dim);
      end;
    W.DimPadded := 0; W.InterPadded := 0;
  finally
    F.Free;
  end;
end;

procedure ConvertAndUploadFp16Weights(M: TK3Model; Verbose: Boolean);
var dim, inter, experts, layers, dp, ip, l, e, i, j: SizeInt;
begin
  dim := M.Config.Dim; inter := M.Config.IntermediateDim; experts := M.Config.NumExperts; layers := M.Config.NumLayers;
  dp := ((dim + 15) div 16) * 16; ip := ((inter + 15) div 16) * 16;
  M.Weights.DimPadded := dp; M.Weights.InterPadded := ip;
  if Verbose then WriteLn(Format('FP16 conversion: dim %d->%d, inter %d->%d', [dim, dp, inter, ip]));
  for l := 0 to layers - 1 do
    with M.Weights.Layers[l] do
    begin
      W1Fp16 := nil; W2Fp16 := nil;
      SetLength(W1Fp16, experts * dp * ip); SetLength(W2Fp16, experts * ip * dp);   { zero == half(0.0) }
      for e := 0 to experts - 1 do
      begin
        for i := 0 to dim - 1 do
          for j := 0 to inter - 1 do
            W1Fp16[e * dp * ip + i * ip + j] := FloatToHalf(W1[e * dim * inter + i * inter + j]);
        for i := 0 to inter - 1 do
          for j := 0 to dim - 1 do
            W2Fp16[e * ip * dp + i * dp + j] := FloatToHalf(W2[e * inter * dim + i * dim + j]);
      end;
    end;
end;

procedure InitModel(M: TK3Model; const Config: TK3Config);
var dim, kv: SizeInt;
begin
  M.Config := Config; dim := Config.Dim;
  SetLength(M.Residual, dim); SetLength(M.AttnOut, dim); SetLength(M.MoeOut, dim);
  SetLength(M.QkvBuffer, 3 * dim); SetLength(M.LnOut, dim);
  SetLength(M.SelectedExperts, Config.TopK); SetLength(M.RoutingWeights, Config.TopK);
  SetLength(M.Logits, Config.VocabSize);
  SetLength(M.ResidualFp16, ((dim + 15) div 16) * 16);
  kv := SizeInt(Config.NumLayers) * Config.MaxSeqLen * dim;
  M.KV.KCache := nil; M.KV.VCache := nil; M.KV.DeltaK := nil;
  SetLength(M.KV.KCache, kv); SetLength(M.KV.VCache, kv); SetLength(M.KV.DeltaK, SizeInt(Config.NumLayers) * dim);
  M.KV.SeqLen := 0;
end;

procedure FreeModel(M: TK3Model);
begin
  M.Residual := nil; M.AttnOut := nil; M.MoeOut := nil; M.QkvBuffer := nil; M.LnOut := nil;
  M.SelectedExperts := nil; M.RoutingWeights := nil; M.Logits := nil; M.ResidualFp16 := nil;
  M.KV.KCache := nil; M.KV.VCache := nil; M.KV.DeltaK := nil;
end;

function CharToToken(C: Byte): LongInt; begin Result := TOK_BYTE_BASE + C; end;

function TokenToChar(Tok: LongInt): AnsiChar;
begin
  if (Tok >= TOK_BYTE_BASE) and (Tok < TOK_BYTE_BASE + 256) then Result := AnsiChar(Tok - TOK_BYTE_BASE)
  else Result := '?';
end;

procedure TokenizePrompt(const Prompt: AnsiString; var Tokens: TIntArray; out NTokens: LongInt; MaxTokens: LongInt);
var i, len: LongInt;
begin
  len := Length(Prompt);
  { strlen() stops at the first NUL }
  if Pos(#0, Prompt) > 0 then len := Pos(#0, Prompt) - 1;
  NTokens := 0;
  Tokens[NTokens] := TOK_BOS; Inc(NTokens);
  i := 0;
  while (i < len) and (NTokens < MaxTokens - 1) do
  begin
    Tokens[NTokens] := CharToToken(Ord(Prompt[i + 1])); Inc(NTokens); Inc(i);
  end;
end;

procedure DetokenizeAndPrint(Token: LongInt);
begin
  if Token = TOK_EOS then Exit;
  Write(TokenToChar(Token));
  Flush(Output);
end;

{ One transformer layer, shared by ForwardToken (MoE path) and DraftStep (dense path). }
procedure ForwardToken(M: TK3Model; TokenId, Pos: LongInt; LogitsOut: PSingle);
var c: ^TK3Config; dim, inter, topK, layers, l, i, blocks, outTiles, wmmaBlocks: LongInt; scale: Single;
    kvOff: SizeInt; L_: ^TK3Layer; AttnIn, HRes: TSingleArray;
begin
  c := @M.Config;
  dim := c^.Dim; inter := c^.IntermediateDim; topK := c^.TopK; layers := c^.NumLayers;
  scale := 1.0 / Single(Sqrt(dim / 32.0));
  blocks := DivCeil(dim, 256);
  embedding_kernel(1, 256, TokenId, Pos, @M.Weights.TokenEmb[0], @M.Weights.PosEmb[0], @M.Residual[0], dim, c^.MaxSeqLen);
  AttnIn := nil; SetLength(AttnIn, dim);
  for l := 0 to layers - 1 do
  begin
    L_ := @M.Weights.Layers[l];
    rmsnorm_kernel(1, 256, @M.Residual[0], @L_^.AttnLn[0], @M.LnOut[0], c^.RmsNormEps, dim);
    qkv_proj_kernel(DivCeil(3 * dim, 256), 256, @M.LnOut[0], @L_^.Qkv[0], @M.QkvBuffer[0], dim);
    kvOff := SizeInt(l) * c^.MaxSeqLen * dim + SizeInt(Pos) * dim;
    Move(M.QkvBuffer[dim], M.KV.KCache[kvOff], dim * SizeOf(Single));
    Move(M.QkvBuffer[2 * dim], M.KV.VCache[kvOff], dim * SizeOf(Single));
    kimi_delta_attention_kernel(blocks, 256, @M.QkvBuffer[0], @M.KV.DeltaK[SizeInt(l) * dim], @M.KV.KCache[kvOff],
      @M.KV.VCache[kvOff], @M.AttnOut[0], scale, dim);
    Move(M.KV.KCache[kvOff], M.KV.DeltaK[SizeInt(l) * dim], dim * SizeOf(Single));
    { :870 passes attn_out as both input and output (racy under CUDA); snapshot the input. }
    Move(M.AttnOut[0], AttnIn[0], dim * SizeOf(Single));
    attn_out_proj_kernel(blocks, 256, @AttnIn[0], @L_^.AttnOut[0], @M.AttnOut[0], dim);
    HRes := Copy(M.Residual, 0, dim);
    for i := 0 to dim - 1 do HRes[i] := HRes[i] + M.AttnOut[i];
    Move(HRes[0], M.Residual[0], dim * SizeOf(Single));
    rmsnorm_kernel(1, 256, @M.Residual[0], @L_^.MoeLn[0], @M.LnOut[0], c^.RmsNormEps, dim);
    latent_moe_router_kernel(1, 32, @M.LnOut[0], @L_^.Router[0], @M.SelectedExperts[0], @M.RoutingWeights[0], dim, c^.NumExperts, topK);
    outTiles := (M.Weights.DimPadded + 15) div 16;
    wmmaBlocks := DivCeil(outTiles * 32, 128);
    convert_fp32_to_fp16_kernel(DivCeil(M.Weights.DimPadded, 256), 256, @M.LnOut[0], @M.ResidualFp16[0], dim, M.Weights.DimPadded);
    wmma_decode_moe_kernel(wmmaBlocks, 128, @M.ResidualFp16[0], @L_^.W1Fp16[0], @L_^.W2Fp16[0], @M.SelectedExperts[0],
      @M.RoutingWeights[0], @M.MoeOut[0], dim, inter, topK, M.Weights.DimPadded, M.Weights.InterPadded);
    for i := 0 to dim - 1 do HRes[i] := HRes[i] + M.MoeOut[i];
    Move(HRes[0], M.Residual[0], dim * SizeOf(Single));
  end;
  rmsnorm_kernel(1, 256, @M.Residual[0], @M.Weights.LnF[0], @M.LnOut[0], c^.RmsNormEps, dim);
  output_head_kernel(DivCeil(c^.VocabSize, 256), 256, @M.LnOut[0], @M.Weights.OutputW[0], @M.Logits[0], c^.VocabSize, dim);
  Move(M.Logits[0], LogitsOut^, c^.VocabSize * SizeOf(Single));
end;

procedure Prefill(M: TK3Model; const Tokens: TIntArray; Len: LongInt);
var i: LongInt; Dummy: TSingleArray;
begin
  WriteLn(Format('Prefilling %d tokens...', [Len]));
  Dummy := nil; SetLength(Dummy, M.Config.VocabSize);
  for i := 0 to Len - 1 do ForwardToken(M, Tokens[i], i, @Dummy[0]);
  M.KV.SeqLen := Len;
end;

procedure LoadDraftModel(const Path: AnsiString; D: TDraftModel; Target: TK3Model);
var F: TFileStream; Magic: array[0..3] of AnsiChar; Version: LongInt; dim, inter, layers, vocab, maxSeq, l: SizeInt; kv: SizeInt;
begin
  F := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    ReadExact(F, Magic, 4); ReadExact(F, Version, 4);
    if Magic <> 'K3D1' then raise EK3Harness.Create('Invalid draft magic');
    ReadExact(F, D.Config, SizeOf(TK3Config));
    D.Config.RmsNormEps := 1e-6;
    dim := D.Config.Dim; inter := D.Config.IntermediateDim; layers := D.Config.NumLayers;
    vocab := D.Config.VocabSize; maxSeq := D.Config.MaxSeqLen;
    D.Target := Target;   { token_emb/pos_emb are the target's (:928-929) }
    ReadFloats(F, D.Weights.LnF, dim); ReadFloats(F, D.Weights.OutputW, vocab * dim);
    D.Weights.Layers := nil; SetLength(D.Weights.Layers, layers);
    for l := 0 to layers - 1 do
      with D.Weights.Layers[l] do
      begin
        ReadFloats(F, AttnLn, dim); ReadFloats(F, Qkv, 3 * dim * dim); ReadFloats(F, AttnOut, dim * dim);
        ReadFloats(F, MoeLn, dim); ReadFloats(F, W1, inter * dim); ReadFloats(F, W2, dim * inter);
      end;
    SetLength(D.Residual, dim); SetLength(D.LnOut, dim); SetLength(D.QkvBuffer, 3 * dim);
    SetLength(D.AttnOut, dim); SetLength(D.FfnOut, dim); SetLength(D.Logits, vocab);
    kv := layers * maxSeq * dim;
    SetLength(D.KV.KCache, kv); SetLength(D.KV.VCache, kv); SetLength(D.KV.DeltaK, layers * dim);
    D.KV.SeqLen := 0;
    WriteLn(Format('Draft: dim=%d layers=%d inter=%d vocab=%d', [dim, layers, inter, vocab]));
  finally
    F.Free;
  end;
end;

function DraftStep(D: TDraftModel; TokenId, Pos: LongInt): LongInt;
var c: ^TK3Config; dim, inter, layers, vocab, l, i, best: LongInt; scale: Single; kvOff: SizeInt; L_: ^TK3Layer;
    AttnIn, HRes: TSingleArray;
begin
  c := @D.Config; dim := c^.Dim; inter := c^.IntermediateDim; layers := c^.NumLayers; vocab := c^.VocabSize;
  scale := 1.0 / Single(Sqrt(dim / 32.0));
  embedding_kernel(1, 256, TokenId, Pos, @D.Target.Weights.TokenEmb[0], @D.Target.Weights.PosEmb[0], @D.Residual[0], dim, c^.MaxSeqLen);
  AttnIn := nil; SetLength(AttnIn, dim);
  for l := 0 to layers - 1 do
  begin
    L_ := @D.Weights.Layers[l];
    rmsnorm_kernel(1, 256, @D.Residual[0], @L_^.AttnLn[0], @D.LnOut[0], c^.RmsNormEps, dim);
    qkv_proj_kernel(DivCeil(3 * dim, 256), 256, @D.LnOut[0], @L_^.Qkv[0], @D.QkvBuffer[0], dim);
    kvOff := SizeInt(l) * c^.MaxSeqLen * dim + SizeInt(Pos) * dim;
    Move(D.QkvBuffer[dim], D.KV.KCache[kvOff], dim * SizeOf(Single));
    Move(D.QkvBuffer[2 * dim], D.KV.VCache[kvOff], dim * SizeOf(Single));
    kimi_delta_attention_kernel(DivCeil(dim, 256), 256, @D.QkvBuffer[0], @D.KV.DeltaK[SizeInt(l) * dim], @D.KV.KCache[kvOff],
      @D.KV.VCache[kvOff], @D.AttnOut[0], scale, dim);
    Move(D.KV.KCache[kvOff], D.KV.DeltaK[SizeInt(l) * dim], dim * SizeOf(Single));
    Move(D.AttnOut[0], AttnIn[0], dim * SizeOf(Single));
    attn_out_proj_kernel(DivCeil(dim, 256), 256, @AttnIn[0], @L_^.AttnOut[0], @D.AttnOut[0], dim);
    HRes := Copy(D.Residual, 0, dim);
    for i := 0 to dim - 1 do HRes[i] := HRes[i] + D.AttnOut[i];
    Move(HRes[0], D.Residual[0], dim * SizeOf(Single));
    rmsnorm_kernel(1, 256, @D.Residual[0], @L_^.MoeLn[0], @D.LnOut[0], c^.RmsNormEps, dim);
    { :1048 passes NULL experts/weights with top_k = 0, so the dense FFN contributes exactly 0. }
    warp_decode_moe_kernel(DivCeil(dim * WARP_SIZE, 128), 128, @D.LnOut[0], @L_^.W1[0], @L_^.W2[0], nil, nil, @D.FfnOut[0], dim, inter, 0);
    for i := 0 to dim - 1 do HRes[i] := HRes[i] + D.FfnOut[i];
    Move(HRes[0], D.Residual[0], dim * SizeOf(Single));
  end;
  rmsnorm_kernel(1, 256, @D.Residual[0], @D.Weights.LnF[0], @D.LnOut[0], c^.RmsNormEps, dim);
  output_head_kernel(DivCeil(vocab, 256), 256, @D.LnOut[0], @D.Weights.OutputW[0], @D.Logits[0], vocab, dim);
  best := 0;
  for i := 1 to vocab - 1 do if D.Logits[i] > D.Logits[best] then best := i;
  D.KV.SeqLen := Pos + 1;
  Result := best;
end;

{ :1076-1118. DEVIATION (documented in PORT_NOTES.md): the CUDA host loop memcpy's the K3Model
  struct itself into h_target_logits (:1088), never fills h_draft_logits, and then hands the
  vocab-sized device buffers target->logits / draft->logits to a kernel that reads
  (gamma+1)*vocab / gamma*vocab floats from them (out-of-bounds). That behaviour is undefined and
  cannot be reproduced. This port fills h_target_logits with the target logits of each verify
  step and h_draft_logits with the draft logits of each draft step, which is what the buffers
  are evidently for. }
procedure SpeculativeDecode(Target: TK3Model; D: TDraftModel; PromptLen, MaxNewTokens: LongInt;
  Temperature, TopP: Single; Gamma: LongInt; Seed: QWord);
var vocab, targetPos, draftPos, token, generated, g, i, acceptCount: LongInt; DraftTokens, Accepted: TIntArray;
    TL, DL: TSingleArray; rngSeed: QWord;
begin
  Assert((Gamma >= 1) and (Gamma <= MAX_GAMMA), 'gamma must be in 1..MAX_GAMMA (h_accepted[MAX_GAMMA], :1095)');
  vocab := Target.Config.VocabSize;
  DraftTokens := nil; Accepted := nil; TL := nil; DL := nil;
  SetLength(DraftTokens, Gamma); SetLength(Accepted, Gamma); SetLength(TL, (Gamma + 1) * vocab); SetLength(DL, Gamma * vocab);
  rngSeed := Seed;   { time(NULL) in the C code; explicit here for reproducibility }
  targetPos := PromptLen; draftPos := PromptLen; token := TOK_BOS; generated := 0;
  while generated < MaxNewTokens do
  begin
    for g := 0 to Gamma - 1 do
    begin
      DraftTokens[g] := DraftStep(D, token, draftPos); Inc(draftPos);
      Move(D.Logits[0], DL[g * vocab], vocab * SizeOf(Single));
    end;
    for g := 0 to Gamma do
      if g = 0 then ForwardToken(Target, token, targetPos + g, @TL[g * vocab])
      else ForwardToken(Target, DraftTokens[g - 1], targetPos + g, @TL[g * vocab]);
    speculative_acceptance_kernel(1, 1, @TL[0], @DL[0], @DraftTokens[0], vocab, Gamma, Temperature, rngSeed,
      @acceptCount, @Accepted[0]);
    Inc(rngSeed);
    i := 0;
    while (i < acceptCount) and (generated < MaxNewTokens) do begin DetokenizeAndPrint(Accepted[i]); Inc(generated); Inc(i); end;
    Inc(targetPos, acceptCount);
    draftPos := targetPos;
    token := Accepted[acceptCount - 1];
  end;
end;

function k3_engine_init(ModelPath: PAnsiChar): Pointer; cdecl;
var M: TK3Model; Cfg: TK3Config; W: TK3Weights;
begin
  M := TK3Model.Create;
  try
    LoadModelWeights(AnsiString(ModelPath), Cfg, W);
    InitModel(M, Cfg);
    M.Weights := W;
    ConvertAndUploadFp16Weights(M);
    Result := M;
  except
    on E: Exception do begin WriteLn(StdErr, E.Message); M.Free; Result := nil; end;   { C code exit(1)s }
  end;
end;

function k3_engine_forward(Engine: Pointer; Token, Pos: LongInt; OutLogits: PSingle): LongInt; cdecl;
begin
  ForwardToken(TK3Model(Engine), Token, Pos, OutLogits);
  Result := 0;
end;

procedure k3_engine_free(Engine: Pointer); cdecl;
begin
  FreeModel(TK3Model(Engine));
  TK3Model(Engine).Free;
end;

end.
