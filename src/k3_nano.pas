{ SPDX-License-Identifier: AGPL-3.0-only
  Port of main() of kernels/k3_nano_harness.cu (:1148-1212).
  Usage: k3_nano <model.bin> <prompt> <max_tokens> <temp> <top_p> [--speculative draft.bin gamma] [--seed N]
  The C code seeds with time(NULL); --seed makes runs reproducible (default: current Unix time). }
{$mode objfpc}{$H+}
program k3_nano;
uses SysUtils, DateUtils, SK_K3Kernels, SK_K3Harness;
var DraftPath: AnsiString; Gamma, i, NTokens, Step, MaxTok, Token: LongInt; Cfg: TK3Config; W: TK3Weights;
    M: TK3Model; D: TDraftModel; Tokens: TIntArray; Logits: TSingleArray; Seed: QWord; T0: TDateTime; Tmp, TopP: Single;
begin
  if ParamCount < 4 then
  begin
    WriteLn(StdErr, 'Usage: ', ParamStr(0), ' <model.bin> <prompt> <max_tokens> <temp> <top_p> [--speculative draft.bin gamma]');
    Halt(1);
  end;
  DraftPath := ''; Gamma := 4; Seed := DateTimeToUnix(Now);
  for i := 6 to ParamCount do
  begin
    if (ParamStr(i) = '--speculative') and (i + 2 <= ParamCount) then begin DraftPath := ParamStr(i + 1); Gamma := StrToInt(ParamStr(i + 2)); end;
    if (ParamStr(i) = '--seed') and (i + 1 <= ParamCount) then Seed := StrToQWord(ParamStr(i + 1));
  end;
  WriteLn('=== Sovereign MiMo-4B Kimi K3 Nano (Free Pascal / pascal-stack port) ==='); WriteLn;
  LoadModelWeights(ParamStr(1), Cfg, W);
  M := TK3Model.Create; InitModel(M, Cfg); M.Weights := W; ConvertAndUploadFp16Weights(M);
  D := nil;
  if DraftPath <> '' then begin D := TDraftModel.Create; LoadDraftModel(DraftPath, D, M); end;
  Tokens := nil; SetLength(Tokens, MAX_SEQ_LEN);
  TokenizePrompt(ParamStr(2), Tokens, NTokens, MAX_SEQ_LEN);
  MaxTok := StrToInt(ParamStr(3)); Tmp := StrToFloat(ParamStr(4)); TopP := StrToFloat(ParamStr(5));
  T0 := Now;
  Prefill(M, Tokens, NTokens);
  if D <> nil then
  begin
    D.KV.SeqLen := NTokens;
    WriteLn; WriteLn(Format('Speculative decoding (gamma=%d)...', [Gamma])); WriteLn;
    SpeculativeDecode(M, D, NTokens, MaxTok, Tmp, TopP, Gamma, Seed);
  end
  else
  begin
    WriteLn; WriteLn(Format('Generating %d tokens...', [MaxTok])); WriteLn;
    Token := TOK_BOS; Logits := nil; SetLength(Logits, Cfg.VocabSize);
    for Step := 0 to MaxTok - 1 do
    begin
      ForwardToken(M, Token, NTokens + Step, @Logits[0]);
      top_p_sample_kernel(1, 1, @M.Logits[0], Cfg.VocabSize, Tmp, TopP, Seed, @Token); Inc(Seed);
      DetokenizeAndPrint(Token);
      if Token = TOK_EOS then Break;
    end;
  end;
  WriteLn; WriteLn; WriteLn(Format('Latency: %.2f ms', [MilliSecondsBetween(Now, T0) * 1.0]));
  D.Free; FreeModel(M); M.Free;
end.
