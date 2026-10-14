# sovereign-kimi → Free Pascal port notes

License: AGPL-3.0-only (see LICENSE). Source: /workspace/sovereign-kimi (unmodified). Local git only; nothing pushed.

## File mapping
| Original | Pascal |
|---|---|
| kimi_butterfly/runtime/threadpiper.rs:12-431 | src/SK_ThreadPiper.pas + threadpiper_table.inc |
| kimi_butterfly/audit/state_audit.rs:5-144 | src/SK_StateAudit.pas + state_audit_table.inc |
| kernel_space .cpp | src/SK_KernelSpace.pas |
| forge_compiler .cpp (1490 passes, literal `\n` emit preserved) | src/SK_ForgeCompiler.pas |
| proof/forge.mlw (Why3) | src/SK_ForgeProof.pas (obligations → Assert, build with -Sa); .mlw kept in proof/ |
| ffi.rs, daemon.rs, lib.rs | src/SK_K3Engine.pas |
| k3_nano_harness .cu (kernels) | src/SK_K3Kernels.pas, SK_Launch.pas (pascal-stack DispatchKernel glue) |
| k3_nano_harness .cu (host / main) | src/SK_K3Harness.pas, src/k3_nano.pas |
| kimi_butterfly.cu, butterfly_factors.cu, .sass | src/SK_Butterfly.pas (incl. SASS interpreter) |
| fused_attention.cu, gemm_online_softmax.cu | src/SK_Attention.pas |
| tensor_pkg, microcode, p45.isa, RTL multiply | src/SK_TensorPkg.pas |
| half/float helpers | src/SK_Half.pas (RNE), SK_FloatOps.pas (correctly-rounded fmaf) |
| *.sv RTL | copied unchanged to original_hw/ (verilator lint run); only host driver logic ported |

## Not ported / not portable
- SystemVerilog: kept as-is by design.
- Mojo: skeleton without compute; only constants ported.
- Python (create_draft_model.py): harness only; inconsistent (43-byte unpack from 40-byte read; 8 `i` with 7 values). Not used as authority.
- Why3 not installed: .mlw not machine-checked; `= by omega` lemma syntax looks invalid.
- CUDA was NOT run on a GPU. CUDA parity is port vs a C++ CPU reference derived from the .cu (refs/).

## pascal-stack model and Alloy guards
DispatchKernel (PascalGPU_Execution.pas:1409) runs threads sequentially with no barriers or shuffles. Kernels with independent threads run per thread. Shuffle, mma and syncthreads kernels run per warp or block using lane arrays.
- Unchecked MatrixAt/Set stride (M:841-875), TensorBatchedMatMul skipping validation (M:2135-2189, 3085), PascalGEMM aliasing (M:1283-1300): Matrix and PascalGEMM are not used. Buffers are bounds-checked dynamic arrays, and disjoint input/output is asserted.
- TDeviceMemory record copies (D:143-149, D:1054): dynamic arrays with a single owner are used instead, so no device records are copied.
- No-op Synchronize (D:1177): nothing relies on it. Dispatch is synchronous and ordering is explicit.
- LaunchGuardCheck enforces MaxThreadsPerBlock=1024. FP exceptions are masked to match CUDA.

## UB choices (identical in port and reference)
These behaviours are undefined in the original, so the port and the reference both make the same choice:
- shared memory starts zeroed;
- cp.async data lands at the wait;
- out-of-bounds row_max/row_denom reads return NaN;
- O_acc starts at 0;
- wmma warps run serially;
- the attn_out_proj aliasing is replaced by a snapshot.

## Original defects found
- fused_attention: V is unused; float2 values are written into a half buffer; reads go past the tiles; the first K tile is never loaded.
- gemm_online_softmax: row_max is read out of bounds; O_acc is uninitialized; P@V uses only k=0.
- Harness:
  - attn_out_proj aliasing;
  - speculative_decode memcpy's the struct and reads out of bounds;
  - tied embeddings upload uninitialized memory;
  - embedding only works for dim ≤ 256;
  - in draft, the FFN uses top_k=0, so its output is 0;
  - wmma has a shared-memory race and lost loads.
- RTL fp16_mul is an integer multiply.
- Microcode: 15 of 25 lines have hex ≠ binary (reported, not raised as an error).

## Build
```
fpc -Mobjfpc -Rintel -O2 -Sa -Fu src -Fu tests -Fi src \
  -Fu /workspace/pascal-stack/handcrafted/core -Fu /workspace/pascal-stack/handcrafted/execution -FU build <program>
./build.sh && ./run_parity.sh
```

## Parity results (real runs of run_parity.sh, 2026-10-14)
- Rust (cargo, original .rs via #[path]) threadpiper/state_audit vs Pascal: 12800 lines identical.
- Rust ffi.rs/daemon.rs (stub engine) vs Pascal: 243 lines identical.
- C++ kernel_space/forge_compiler (original .cpp #included) vs Pascal: 9452 lines identical.
- CUDA k3_nano_harness port vs CPU reference: 8313 values across 20 tests, 0 mismatches, bit-exact.
- CUDA butterfly, SASS and attention port vs reference: 56592 values, bit-exact except gemm_online_softmax.
  - gemm_online_softmax: 6 values differ, max rel 4.81e-6 and max abs 6.5e-11, inside the stated tolerance of rel 1e-5. Most of that output is NaN because of UB in the original.
- Self-checks (Why3 Asserts and others): 2337 passed, 0 failed.
