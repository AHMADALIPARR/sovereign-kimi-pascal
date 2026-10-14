{ SPDX-License-Identifier: AGPL-3.0-only
  Ports onto pascal-stack of:
   * kimi_butterfly/cuda/kimi_butterfly.cu:1-31   (threadpiper_router, gated_delta_butterfly, triple_lock_kernel)
   * kimi_butterfly/circuits/butterfly_factors.cu  (device factor functions, :3-287)
   * kimi_butterfly/sass/butterfly.sass:1-25       (the annotated pseudo-SASS program, as a per-warp
                                                   register-machine interpreter of exactly those 20 instructions)
  The .cu files carry 219 ("CUDA factor 001..219", kimi_butterfly.cu:32-250) and the .sass 95
  ("SASS annotation", butterfly.sass:26-120) comment-only lines with no semantics. }
{$mode objfpc}{$H+}{$Q-}{$R-}{$POINTERMATH ON}
unit SK_Butterfly;

interface

uses SysUtils, PascalGPU_Types, SK_Launch, SK_FloatOps;

const
  WARP = 32; EXPERTS = 896; ACTIVE = 16;   { kimi_butterfly.cu:3-5 }
  WF = 32;                                  { butterfly_factors.cu:3 }
  FACTOR_COUNT = 260;                       { butterfly_factors.cu:27-286 }

type
  TWarpF = array[0..31] of Single;
  TWarpU = array[0..31] of LongWord;

{ ---- kimi_butterfly.cu kernels ---- }
procedure threadpiper_router(Grid, Block: LongWord; X, RW: PSingle; Ids: PLongInt; Weights: PSingle; Dim: LongInt);   { :11-18 }
procedure gated_delta_butterfly(Grid, Block: LongWord; X, Prev, Gate: PSingle; Ids: PLongInt; Weights, State: PSingle; Dim: LongInt); { :19-25 }
procedure triple_lock_kernel(Grid, Block: LongWord; State: PSingle; Seal, OutP: PLongWord; N: LongInt);                 { :26-31 }
procedure BSum(var V: TWarpF);   { :6-10, xor offsets 16,8,4,2,1 }

{ ---- butterfly_factors.cu device functions (lane-array form) ---- }
procedure FactorBfly(var V: TWarpF);                         { :7-11, offsets 1,2,4,8,16 }
function FactorGate(X, G: Single): Single; inline;           { :12 }
function FactorDelta(X, P: Single): Single; inline;          { :13 }
function FactorRecur(S, D, G: Single): Single; inline;       { :14 }
function FactorLock(X, Y, Z: LongWord): LongWord; inline;    { :15 }
function LaneOf(ThreadIdxX: LongWord): LongInt; inline;      { :16 }
function WarpOf(ThreadIdxX: LongWord): LongInt; inline;      { :17 }
function Route(X, W: Single): Single; inline;                { :18 }
procedure Reduce16(var V: TWarpF);                           { :19-21 }
procedure RecursiveReduce(var V: TWarpF; Stage: LongInt);    { :22-26 }
function Factor(N: Integer; X, Y: Single): Single;           { :27-286 factor_NNN = fmaf(x,y,x) }

{ ---- butterfly.sass ---- }
type
  TSassLane = record R: array[0..31] of LongWord; end;   { R0..R31 raw 32-bit registers; RZ handled as 0 }
  TSassWarp = array[0..31] of TSassLane;
  PSassWarpArray = ^TSassWarp;
{ Executes butterfly.sass:3-25 for one warp. Mem is the global memory (32-bit words) addressed by
  R20 (LDG source) and R21 (STG destination) as word indices. TidBase/CtaId feed SR_TID.X/SR_CTAID.X. }
procedure RunButterflySass(var W: TSassWarp; Mem: PLongWord; TidBase, CtaId: LongWord);
procedure butterfly_sass_kernel(Grid, Block: LongWord; Regs: PSassWarpArray; Mem: PLongWord);


implementation

procedure ShflXorAdd(var V: TWarpF; M: Integer);
var t: TWarpF; l: Integer;
begin
  for l := 0 to 31 do t[l] := V[l xor M];
  for l := 0 to 31 do V[l] := V[l] + t[l];
end;

procedure BSum(var V: TWarpF);
var m: Integer;
begin
  m := 16;
  while m <> 0 do begin ShflXorAdd(V, m); m := m shr 1; end;
end;

procedure FactorBfly(var V: TWarpF);
var m: Integer;
begin
  m := 1;
  while m < WF do begin ShflXorAdd(V, m); m := m shl 1; end;
end;

function FactorGate(X, G: Single): Single; inline; begin Result := X * G; end;
function FactorDelta(X, P: Single): Single; inline; begin Result := X - P; end;
function FactorRecur(S, D, G: Single): Single; inline; begin Result := S + FactorGate(D, G); end;
function FactorLock(X, Y, Z: LongWord): LongWord; inline; begin Result := (X xor Y) xor Z; end;
function LaneOf(ThreadIdxX: LongWord): LongInt; inline; begin Result := ThreadIdxX and 31; end;
function WarpOf(ThreadIdxX: LongWord): LongInt; inline; begin Result := ThreadIdxX shr 5; end;
function Route(X, W: Single): Single; inline; begin Result := X * W; end;

procedure Reduce16(var V: TWarpF);
begin ShflXorAdd(V, 16); ShflXorAdd(V, 8); ShflXorAdd(V, 4); ShflXorAdd(V, 2); ShflXorAdd(V, 1); end;

procedure RecursiveReduce(var V: TWarpF; Stage: LongInt);
begin
  if Stage <= 0 then Exit;
  ShflXorAdd(V, 1 shl ((Stage - 1) and 4));   { (stage-1)&4 is 0 or 4 -> offsets 1 or 16 }
  RecursiveReduce(V, Stage - 1);
end;

function Factor(N: Integer; X, Y: Single): Single;
begin
  Assert((N >= 1) and (N <= FACTOR_COUNT), 'factor index out of range (001..260)');
  Result := FmaF(X, Y, X);
end;

{ ---- threadpiper_router: warp granularity ---- }
type
  TRouterArgs = record X, RW: PSingle; Ids: PLongInt; W: PSingle; Dim: LongInt; CudaBlock: LongWord; end;
  PRouterArgs = ^TRouterArgs;

procedure RouterBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PRouterArgs; lane, tid, e, d, idx: LongInt; s: TWarpF;
begin
  A := PRouterArgs(Data);
  tid := T.X * 32;
  if tid >= EXPERTS then Exit;   { whole warp exits (EXPERTS is a multiple of 32) }
  for lane := 0 to 31 do
  begin
    e := tid + lane; s[lane] := 0.0; d := 0;
    while d < A^.Dim do
    begin
      idx := (d + e) mod A^.Dim;
      s[lane] := FmaF(A^.X[idx], A^.RW[PtrInt(e) * A^.Dim + idx], s[lane]);
      Inc(d, WARP);
    end;
  end;
  BSum(s);
  for lane := 0 to 31 do
    if tid + lane < ACTIVE then begin A^.Ids[tid + lane] := tid + lane; A^.W[tid + lane] := s[lane]; end;
end;

procedure threadpiper_router(Grid, Block: LongWord; X, RW: PSingle; Ids: PLongInt; Weights: PSingle; Dim: LongInt);
var A: TRouterArgs;
begin
  Assert(Block mod 32 = 0, 'threadpiper_router: partial warps would make the full-mask shuffle undefined');
  A.X := X; A.RW := RW; A.Ids := Ids; A.W := Weights; A.Dim := Dim; A.CudaBlock := Block;
  Launch1D('threadpiper_router', Grid, Block div 32, 0, @RouterBody, @A);
end;

{ ---- gated_delta_butterfly: warp granularity ---- }
type
  TGdbArgs = record X, Prev, Gate: PSingle; Ids: PLongInt; W, State: PSingle; Dim: LongInt; CudaBlock: LongWord; end;
  PGdbArgs = ^TGdbArgs;

procedure GdbBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PGdbArgs; lane, i0, i, tx: LongInt; acc: TWarpF; d, g: Single; live: Integer;
begin
  A := PGdbArgs(Data);
  i0 := LongInt(B.X * A^.CudaBlock + T.X * 32);
  live := 0;
  for lane := 0 to 31 do if i0 + lane < A^.Dim then Inc(live);
  if live = 0 then Exit;
  Assert(live = 32, 'gated_delta_butterfly: partially populated warp reaches a full-mask __shfl_xor_sync (undefined in CUDA)');
  for lane := 0 to 31 do
  begin
    i := i0 + lane; tx := LongInt(T.X * 32) + lane;
    d := A^.X[i] - A^.Prev[i]; g := A^.Gate[i];
    acc[lane] := d * g;
    acc[lane] := acc[lane] * A^.W[A^.Ids[tx mod ACTIVE]];
  end;
  BSum(acc);
  for lane := 0 to 31 do A^.State[i0 + lane] := acc[lane];
end;

procedure gated_delta_butterfly(Grid, Block: LongWord; X, Prev, Gate: PSingle; Ids: PLongInt; Weights, State: PSingle; Dim: LongInt);
var A: TGdbArgs;
begin
  Assert(Block mod 32 = 0);
  A.X := X; A.Prev := Prev; A.Gate := Gate; A.Ids := Ids; A.W := Weights; A.State := State; A.Dim := Dim; A.CudaBlock := Block;
  Launch1D('gated_delta_butterfly', Grid, Block div 32, 0, @GdbBody, @A);
end;

{ ---- triple_lock_kernel: thread granularity ---- }
type
  TLockArgs = record State: PSingle; Seal, OutP: PLongWord; N: LongInt; end;
  PLockArgs = ^TLockArgs;

procedure LockBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PLockArgs; i: LongInt; x, la, lb, lc: LongWord;
begin
  A := PLockArgs(Data);
  i := LongInt(B.X * BD.X + T.X); if i >= A^.N then Exit;
  x := FloatAsUInt(A^.State[i]);
  la := x xor A^.Seal[i];
  lb := (la shl 13) or (la shr 19);
  lc := lb xor LongWord(x * LongWord($9e3779b9));
  A^.OutP[i] := lc;
end;

procedure triple_lock_kernel(Grid, Block: LongWord; State: PSingle; Seal, OutP: PLongWord; N: LongInt);
var A: TLockArgs;
begin
  A.State := State; A.Seal := Seal; A.OutP := OutP; A.N := N;
  Launch1D('triple_lock_kernel', Grid, Block, 0, @LockBody, @A);
end;

{ ---- butterfly.sass interpreter ---- }
procedure RunButterflySass(var W: TSassWarp; Mem: PLongWord; TidBase, CtaId: LongWord);
var l, m: Integer; t: array[0..31] of LongWord;
  function F(U: LongWord): Single; inline; begin Result := UIntAsFloat(U); end;
  function U(X: Single): LongWord; inline; begin Result := FloatAsUInt(X); end;
begin
  for l := 0 to 31 do
  begin
    W[l].R[0] := TidBase + LongWord(l);                     { :3  S2R R0, SR_TID.X }
    W[l].R[1] := CtaId;                                     { :4  S2R R1, SR_CTAID.X }
    W[l].R[2] := W[l].R[0] xor W[l].R[1] xor 0;             { :5  LOP3 R2, R0, R1, RZ, 0x96 (a^b^c) }
  end;
  m := 16;
  while m >= 1 do                                           { :7-16 SHFL.XOR R8,R8,m,0x1f ; FADD R9,R9,R8 }
  begin
    for l := 0 to 31 do t[l] := W[l xor m].R[8];
    for l := 0 to 31 do W[l].R[8] := t[l];
    for l := 0 to 31 do W[l].R[9] := U(F(W[l].R[9]) + F(W[l].R[8]));
    m := m shr 1;
  end;
  for l := 0 to 31 do
  begin
    W[l].R[10] := U(F(W[l].R[6]) + (-F(W[l].R[7])));       { :18 FADD R10, R6, -R7 }
    W[l].R[10] := U(F(W[l].R[10]) * F(W[l].R[11]));        { :19 FMUL R10, R10, R11 }
    W[l].R[12] := Mem[W[l].R[20]];                          { :21 LDG.E R12, [R20] }
  end;
  for l := 0 to 31 do Mem[W[l].R[21]] := W[l].R[12];        { :22 STG.E [R21], R12 }
  for l := 0 to 31 do
    W[l].R[13] := W[l].R[12] xor W[l].R[14] xor W[l].R[15]; { :24 LOP3 R13, R12, R14, R15, 0x96 }
  { :25 EXIT }
end;

type
  TSassArgs = record Regs: PSassWarpArray; Mem: PLongWord; CudaBlock: LongWord; end;
  PSassArgs = ^TSassArgs;

procedure SassBody(const T, B, BD, GD: TDim3D; Data: Pointer);
var A: PSassArgs; wi: LongWord;
begin
  A := PSassArgs(Data);
  wi := B.X * (A^.CudaBlock div 32) + T.X;
  RunButterflySass(A^.Regs[wi], A^.Mem, T.X * 32, B.X);
end;

procedure butterfly_sass_kernel(Grid, Block: LongWord; Regs: PSassWarpArray; Mem: PLongWord);
var A: TSassArgs;
begin
  Assert(Block mod 32 = 0);
  A.Regs := Regs; A.Mem := Mem; A.CudaBlock := Block;
  Launch1D('butterfly_sass', Grid, Block div 32, 0, @SassBody, @A);
end;

end.
