{ SPDX-License-Identifier: AGPL-3.0-only
  Glue between the ported CUDA kernels and pascal-stack's device/kernel/launch model
  (PascalGPU_Types: TDim3D/TLaunchConfig; PascalGPU_Execution: IKernelFunction, TKernelDescriptor,
  TExecutionScheduler, DispatchKernel, LaunchGuardCheck).

  pascal-stack's DispatchKernel (handcrafted/execution/PascalGPU_Execution.pas:1409-1490) runs every
  (block, thread) pair sequentially and has no barrier or warp-shuffle primitive. Therefore:
   * kernels with independent threads are launched at THREAD granularity (BlockDim = CUDA blockDim);
   * kernels using __syncthreads / __shfl_xor_sync / mma / wmma are launched at BLOCK or WARP
     granularity: one pascal-stack "thread" executes one whole CUDA block (or warp) in lock-step
     phases with explicit per-lane arrays. This preserves the CUDA program order at each barrier.
  Synchronize in the simulator is a no-op (pascal-stack device D:1177), so ordering is guaranteed by
  DispatchKernel being synchronous, not by any sync call. }
{$mode objfpc}{$H+}
unit SK_Launch;

interface

uses SysUtils, Math, PascalGPU_Types, PascalGPU_Execution;

type
  TKernelProc = procedure(const T, B, BD, GD: TDim3D; Data: Pointer);
  EKernelLaunch = class(Exception);

{ Launch Proc over Grid x Block (1-D). Raises on any non-success TResult. }
procedure Launch1D(const Name: AnsiString; GridX, BlockX, SharedMem: LongWord; Proc: TKernelProc; Data: Pointer);
function LaunchCount: QWord;

implementation

type
  TProcKernel = class(TInterfacedObject, IKernelFunction)
    FProc: TKernelProc;
    constructor Create(P: TKernelProc);
    procedure Execute(const AThreadIdx: TThreadIdx; const ABlockIdx: TBlockIdx;
                      const ABlockDim: TBlockDim; const AGridDim: TGridDim; AUserData: Pointer);
  end;

var
  GSched: TExecutionScheduler;
  GInit: Boolean = False;
  GProps: TDeviceProperties;
  GCount: QWord = 0;

constructor TProcKernel.Create(P: TKernelProc);
begin inherited Create; FProc := P; end;

procedure TProcKernel.Execute(const AThreadIdx: TThreadIdx; const ABlockIdx: TBlockIdx;
  const ABlockDim: TBlockDim; const AGridDim: TGridDim; AUserData: Pointer);
begin
  FProc(AThreadIdx, ABlockIdx, ABlockDim, AGridDim, AUserData);
end;

procedure EnsureInit;
begin
  if GInit then Exit;
  if InitScheduler(GSched, 1) <> PGPU_SUCCESS then raise EKernelLaunch.Create('InitScheduler failed');
  FillChar(GProps, SizeOf(GProps), 0);
  GProps.MaxThreadsPerBlock := 1024;
  GProps.MaxSharedMemPerBlock := 166 * 1024;   { sm_80/sm_86-class limit used by the .cu launchers }
  GProps.MaxRegistersPerBlock := 65536;
  GProps.WarpSize := 32;
  GProps.MultiProcessorCount := 1;
  GProps.MaxGridDim := MakeDim3D($7FFFFFFF, 65535, 65535);
  GProps.MaxBlockDim := MakeDim3D(1024, 1024, 64);
  GInit := True;
end;

procedure Launch1D(const Name: AnsiString; GridX, BlockX, SharedMem: LongWord; Proc: TKernelProc; Data: Pointer);
var KD: TKernelDescriptor; R: TResult;
begin
  EnsureInit;
  if (GridX = 0) or (BlockX = 0) then Exit;   { CUDA: zero-sized launch is an error that launches nothing }
  KD.Name := Name;
  KD.Config := MakeLaunchConfig1D(GridX, BlockX, SharedMem, 0);
  KD.KernelFunc := TProcKernel.Create(Proc);
  KD.UserData := Data;
  R := LaunchGuardCheck(KD, GProps);
  if R <> PGPU_SUCCESS then raise EKernelLaunch.CreateFmt('LaunchGuardCheck rejected %s (code %d)', [Name, R]);
  R := DispatchKernel(GSched, KD);
  if R <> PGPU_SUCCESS then raise EKernelLaunch.CreateFmt('DispatchKernel failed for %s (code %d)', [Name, R]);
  Inc(GCount);
end;

function LaunchCount: QWord; begin Result := GCount; end;

initialization
  { CUDA arithmetic is IEEE non-trapping: inf/NaN propagate instead of raising. }
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow, exUnderflow, exPrecision]);
finalization
  if GInit then DestroyScheduler(GSched);
end.
