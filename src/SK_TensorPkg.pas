{ SPDX-License-Identifier: AGPL-3.0-only
  Host-side constants for the RTL (rtl/tensor_pkg.sv:9-32; the RTL itself is kept verbatim under
  original_hw/) plus host-side loaders for the two data files the repo ships:
   * microcode/extended_ws_os_tc.txt  ("AAAA: HHHHHHHH | BBBB...B" lines)
   * kimi_butterfly/p45/p45.isa       (ISA header, OP list, RULE n CHECK REGISTER_BOUNDS)
  The repository contains no host driver for either file; these loaders only parse and validate what the
  files literally state (whether the hex word equals the binary word on each line, contiguous rule numbering). No semantics are assigned to
  microcode words beyond that, because none are defined anywhere in the sources. }
{$mode objfpc}{$H+}
unit SK_TensorPkg;

interface

uses SysUtils, Classes;

type
  TDataflow = (DF_WEIGHT_STATIONARY = 0, DF_OUTPUT_STATIONARY = 1, DF_INPUT_STATIONARY = 2, DF_TENSOR_CORE = 3);
const
  OP_NOP = $0; OP_LOAD = $1; OP_STORE = $2; OP_ADD = $3; OP_MUL = $4; OP_MMA = $5; OP_MAX = $6; OP_EXP = $7;
  OP_SCALE = $8; OP_REDUCE = $9; OP_SHFL = $A; OP_MOV = $B; OP_CMP = $C; OP_BR = $D; OP_SYNC = $E; OP_HALT = $F;
  { rtl/tensor_core_fp16.sv:9-13 default parameters }
  TC_M = 16; TC_N = 8; TC_K = 16;

type
  TMicroWord = record Addr: LongWord; Word: LongWord; BinWord: LongWord; Consistent: Boolean; Line: Integer; end;
  TMicrocode = array of TMicroWord;
  TP45Isa = record
    Name: AnsiString; Regs, Warp, Experts, Active: LongInt;
    Ops: array of AnsiString;
    RuleCount: Integer;
  end;
  EMicrocode = class(Exception);

function LoadMicrocode(const Path: AnsiString): TMicrocode;
function LoadP45Isa(const Path: AnsiString): TP45Isa;
{ rtl/tensor_core_fp16.sv:30-32 fp16_mul_fp32 is a 16x16 SIGNED INTEGER product truncated to 32 bits
  (despite its name). Host-side model of that function, and of one full start..done MAC sequence (:34-70). }
function Fp16MulFp32Rtl(A, B: Word): LongWord;

implementation

function Fp16MulFp32Rtl(A, B: Word): LongWord;
begin
  Result := LongWord(LongInt(SmallInt(A)) * LongInt(SmallInt(B)));
end;

function LoadMicrocode(const Path: AnsiString): TMicrocode;
var L: TStringList; i, n, p, bar: Integer; s, a, h, b: AnsiString; hv, bv: LongWord; j: Integer;
begin
  Result := nil; L := TStringList.Create;
  try
    L.LoadFromFile(Path); n := 0;
    for i := 0 to L.Count - 1 do
    begin
      s := Trim(L[i]);
      if (s = '') or (s[1] = '#') then Continue;
      p := Pos(':', s); bar := Pos('|', s);
      if (p = 0) or (bar = 0) then raise EMicrocode.CreateFmt('%s:%d: malformed line', [Path, i + 1]);
      a := Trim(Copy(s, 1, p - 1)); h := Trim(Copy(s, p + 1, bar - p - 1)); b := Trim(Copy(s, bar + 1, MaxInt));
      if (Length(h) <> 8) or (Length(b) <> 32) then raise EMicrocode.CreateFmt('%s:%d: bad field width', [Path, i + 1]);
      hv := StrToInt('$' + h); bv := 0;
      for j := 1 to 32 do
        case b[j] of '0': bv := bv shl 1; '1': bv := (bv shl 1) or 1;
        else raise EMicrocode.CreateFmt('%s:%d: non-binary digit', [Path, i + 1]); end;
      SetLength(Result, n + 1);
      Result[n].Addr := StrToInt('$' + a); Result[n].Word := hv; Result[n].BinWord := bv; Result[n].Consistent := hv = bv; Result[n].Line := i + 1; Inc(n);
    end;
  finally
    L.Free;
  end;
end;

function LoadP45Isa(const Path: AnsiString): TP45Isa;
var L: TStringList; i: Integer; s: AnsiString; parts: TStringArray; n: Integer;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(Path);
    Result.Name := ''; Result.Regs := 0; Result.Warp := 0; Result.Experts := 0; Result.Active := 0; Result.Ops := nil; Result.RuleCount := 0;
    for i := 0 to L.Count - 1 do
    begin
      s := Trim(L[i]);
      if (s = '') or (s[1] = ';') then Continue;
      parts := s.Split([' ']);
      if parts[0] = 'ISA' then Result.Name := parts[1]
      else if parts[0] = 'REGS' then Result.Regs := StrToInt(parts[1])
      else if parts[0] = 'WARP' then Result.Warp := StrToInt(parts[1])
      else if parts[0] = 'EXPERTS' then Result.Experts := StrToInt(parts[1])
      else if parts[0] = 'ACTIVE' then Result.Active := StrToInt(parts[1])
      else if parts[0] = 'OP' then begin n := Length(Result.Ops); SetLength(Result.Ops, n + 1); Result.Ops[n] := parts[1]; end
      else if parts[0] = 'RULE' then
      begin
        Inc(Result.RuleCount);
        if (StrToInt(parts[1]) <> Result.RuleCount) or (parts[2] <> 'CHECK') or (parts[3] <> 'REGISTER_BOUNDS') then
          raise EMicrocode.CreateFmt('%s:%d: unexpected rule line', [Path, i + 1]);
      end
      else raise EMicrocode.CreateFmt('%s:%d: unknown directive %s', [Path, i + 1, parts[0]]);
    end;
  finally
    L.Free;
  end;
end;

end.
