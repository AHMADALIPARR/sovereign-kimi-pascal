{ SPDX-License-Identifier: AGPL-3.0-only
  Port of kimi_butterfly/compiler/forge_compiler.cpp (namespace forge). }
{$mode objfpc}{$H+}{$Q-}{$R-}
unit SK_ForgeCompiler;

interface

uses SysUtils, Generics.Collections;

type
  { forge_compiler.cpp:8 }
  TKind = (kModule, kBlock, kLet, kIf, kLoop, kCall, kBinary, kConstant, kLoad, kStore,
           kButterfly, kGate, kDelta, kRecur, kRoute, kSeal);
  { forge_compiler.cpp:9 }
  TFType = (tI1, tI8, tI16, tI32, tI64, tF16, tBF16, tF32, tF8, tF4, tPtr, tVoid);
  TSpan = record BeginPos, EndPos: LongWord; end;                          { :10 }
  TNode = record                                                           { :11 }
    Kind: TKind; Typ: TFType; Span: TSpan; Name: AnsiString; Imm: Int64;
    Args: array of LongWord;
  end;
  TAst = record Nodes: array of TNode; Root: LongWord; end;               { :12 }
  TInstr = record Op, Dst, A, B: Word; Imm: LongInt; end;                  { :13 }
  TIR = record Code: array of TInstr; Regs: LongWord; end;                 { :14 }
  TTarget = record Sm, RegsPerThread, Warp, L2Kb: LongWord; Fp4, Fp8, Tma, Dsm: Boolean; end; { :15 }
  TDiagnostic = record Span: TSpan; Text: AnsiString; end;                 { :16 }
  TForgeContext = record                                                   { :17 }
    Target: TTarget;
    Errors: array of TDiagnostic;
    Symbols: specialize TDictionary<AnsiString, LongWord>; { declared in :17, never used by any function }
  end;

const
  PASS_COUNT = 1490;   { forge_compiler.cpp:21-1510 }

function Compile(var C: TForgeContext; const A: TAst): TIR;   { :1518 }
function EmitP45(const IR: TIR): AnsiString;                   { :1519 }
function EmitSass(const IR: TIR): AnsiString;                  { :1520 }
{ pass_0001..pass_1490 (:21-1510) are `static` and never called by compile();
  they are identical: fail "unsupported warp contract" when target.warp<>32. }
procedure RunPass(N: Integer; var C: TForgeContext; var IR: TIR; const A: TAst);

implementation

procedure Fail(var C: TForgeContext; const S: TSpan; const M: AnsiString);   { :18 }
var n: SizeInt;
begin
  n := Length(C.Errors);
  SetLength(C.Errors, n + 1);
  C.Errors[n].Span := S; C.Errors[n].Text := M;
end;

function AddReg(var IR: TIR): LongWord;   { :19 — post-increment }
begin
  Result := IR.Regs; Inc(IR.Regs);
end;

function OpOf(K: TKind): Word;   { :20 }
begin
  Result := Word(Ord(K));
end;

procedure Push(var IR: TIR; Op, Dst, A, B: Word; Imm: LongInt);
var n: SizeInt;
begin
  n := Length(IR.Code); SetLength(IR.Code, n + 1);
  IR.Code[n].Op := Op; IR.Code[n].Dst := Dst; IR.Code[n].A := A; IR.Code[n].B := B; IR.Code[n].Imm := Imm;
end;

procedure LowerNode(var C: TForgeContext; var IR: TIR; const N: TNode);   { :1511-1517 }
var r: LongWord;
begin
  case N.Kind of
    kButterfly:
      begin r := AddReg(IR); Push(IR, OpOf(N.Kind), Word(r), Word(Length(N.Args)), 0, 16); end;
    kGate, kDelta, kRecur, kRoute, kSeal:
      begin r := AddReg(IR); Push(IR, OpOf(N.Kind), Word(r), 0, 0, LongInt(N.Imm)); end;
  else
    begin r := AddReg(IR); Push(IR, OpOf(N.Kind), Word(r), 0, 0, LongInt(N.Imm)); end;
  end;
end;

function Compile(var C: TForgeContext; const A: TAst): TIR;
var i: SizeInt; Empty: TSpan;
begin
  Result.Code := nil; Result.Regs := 0;
  for i := 0 to High(A.Nodes) do LowerNode(C, Result, A.Nodes[i]);
  if Result.Regs > C.Target.RegsPerThread then
  begin
    Empty.BeginPos := 0; Empty.EndPos := 0;
    Fail(C, Empty, 'register contract exceeded');
  end;
end;

{ NB: the C++ source writes "\\n", i.e. a literal backslash followed by 'n', not a newline. }
function EmitP45(const IR: TIR): AnsiString;
var i: SizeInt;
begin
  Result := '';
  for i := 0 to High(IR.Code) do
    Result := Result + 'P45.' + IntToStr(IR.Code[i].Op) + ' R' + IntToStr(IR.Code[i].Dst) + '\n';
end;

function EmitSass(const IR: TIR): AnsiString;
var i: SizeInt;
begin
  Result := '';
  for i := 0 to High(IR.Code) do
    Result := Result + 'OP ' + IntToStr(IR.Code[i].Op) + ' R' + IntToStr(IR.Code[i].Dst) + '\n';
end;

procedure RunPass(N: Integer; var C: TForgeContext; var IR: TIR; const A: TAst);
var Empty: TSpan;
begin
  Assert((N >= 1) and (N <= PASS_COUNT), 'pass index out of range (0001..1490)');
  { `if(a.nodes.size()>0){ ir.regs = ir.regs; }` is a no-op }
  if C.Target.Warp <> 32 then
  begin
    Empty.BeginPos := 0; Empty.EndPos := 0;
    Fail(C, Empty, 'unsupported warp contract');
  end;
end;

end.
