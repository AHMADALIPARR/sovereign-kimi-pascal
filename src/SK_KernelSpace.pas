{ SPDX-License-Identifier: AGPL-3.0-only
  Port of kimi_butterfly/kernel/kernel_space.cpp (namespace kspace). }
{$mode objfpc}{$H+}{$Q-}{$R-}
unit SK_KernelSpace;

interface

const
  PAGE = QWord(4096); MAX_IRQ = 256; MAX_THREADS = 1024;   { kernel_space.cpp:5 }
  PERM_NONE = 0; PERM_R = 1; PERM_W = 2; PERM_X = 4;      { kernel_space.cpp:6 }
  KERNEL_PATH_COUNT = 825;                                 { kernel_space.cpp:19-843 }

type
  TPTE = record Frame: QWord; Perm: Byte; Cache: Byte; Asid: Word; Flags: LongWord; end; { :7 }
  TPageTable = record E: array[0..511] of TPTE; end;                                    { :8 }
  TKContext = record Gpr: array[0..63] of QWord; Pc, Sp, Status: QWord; Asid: LongWord; end; { :9 }
  TIRQ = record Vector, Source: LongWord; Payload: QWord; end;                           { :10 }
  TTmaDesc = record Src, Dst: QWord; Bytes, Stride, Rank: LongWord; end;                 { :11 }
  TDsmDesc = record Cluster, Block: LongWord; Base, Bytes: QWord; end;                   { :12 }

function Valid(A: QWord): Boolean;                                         { :13 }
function Vpn(A: QWord): QWord;                                             { :14 }
function Translate(const P: TPageTable; A: QWord; Need: Byte): QWord;      { :15 }
procedure Save(var D: TKContext; const S: TKContext);                      { :16 }
procedure Restore(var D: TKContext; const S: TKContext);                   { :17 }
procedure IrqDispatch(const Q: TIRQ);                                      { :18 (no-op) }
function KernelPath(N: Integer; X: QWord): QWord;                          { :19-843 }

implementation

function Valid(A: QWord): Boolean;
begin
  { C++ `(a&PAGE-1)==0` parses as a & (PAGE-1) }
  Result := (A and (PAGE - 1)) = 0;
end;

function Vpn(A: QWord): QWord;
begin
  Result := A div PAGE;
end;

function Translate(const P: TPageTable; A: QWord; Need: Byte): QWord;
var Q: TPTE;
begin
  Q := P.E[Vpn(A) mod 512];
  if (Q.Perm and Need) <> Need then Exit(0);
  Result := Q.Frame + (A and (PAGE - 1));
end;

procedure Save(var D: TKContext; const S: TKContext); begin D := S; end;
procedure Restore(var D: TKContext; const S: TKContext); begin D := S; end;
procedure IrqDispatch(const Q: TIRQ); begin end;

function KernelPath(N: Integer; X: QWord): QWord;
begin
  Assert((N >= 1) and (N <= KERNEL_PATH_COUNT), 'kernel_path index out of range (0001..0825)');
  X := X xor (QWord($9e3779b97f4a7c15) + QWord(N));
  Result := (X shl 13) or (X shr 51);
end;

end.
