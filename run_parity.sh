#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Builds the originals (Rust via cargo, C++ via g++) and the CPU references for the CUDA kernels, runs
# everything on identical inputs and compares. CUDA itself is NOT run (no GPU/nvcc on this machine).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); B=$ROOT/build
"$ROOT/build.sh" >/dev/null
CXX="g++ -std=c++20 -O2 -ffp-contract=off -w"
(cd "$ROOT/tests/parity_rust" && cargo build --release --offline -q)
(cd "$ROOT/tests/parity_ffi"  && cargo build --release --offline -q)
$CXX "$ROOT/tests/parity_cpp/parity_cpp.cpp" -o "$B/parity_cpp"
$CXX "$ROOT/refs/k3_ref.cpp" -o "$B/k3_ref"
$CXX "$ROOT/refs/bfly_attn_ref.cpp" -o "$B/bfly_attn_ref"
$CXX "$ROOT/refs/make_model.cpp" -o "$B/make_model"
$CXX "$ROOT/refs/compare.cpp" -o "$B/compare"
cd "$ROOT/tests"; "$B/make_model"
echo "== Rust originals (threadpiper.rs, state_audit.rs) vs Pascal: exact text diff"
"$ROOT/tests/parity_rust/target/release/parity_rust" 200 > "$B/rust_orig.txt"; "$B/parity_rust_pas" 200 > "$B/rust_pas.txt"
cmp "$B/rust_orig.txt" "$B/rust_pas.txt" && echo "IDENTICAL ($(wc -l < "$B/rust_orig.txt") lines)"
echo "== Rust originals (ffi.rs, daemon.rs; stub engine) vs Pascal: exact text diff"
"$ROOT/tests/parity_ffi/target/release/parity_ffi" > "$B/ffi_orig.txt" 2>/dev/null; "$B/parity_ffi_pas" > "$B/ffi_pas.txt" 2>/dev/null
cmp "$B/ffi_orig.txt" "$B/ffi_pas.txt" && echo "IDENTICAL ($(wc -l < "$B/ffi_orig.txt") lines)"
echo "== C++ originals (kernel_space.cpp, forge_compiler.cpp) vs Pascal: exact text diff"
"$B/parity_cpp" 200 > "$B/cpp_orig.txt"; "$B/parity_cpp_pas" 200 > "$B/cpp_pas.txt"
cmp "$B/cpp_orig.txt" "$B/cpp_pas.txt" && echo "IDENTICAL ($(wc -l < "$B/cpp_orig.txt") lines)"
echo "== CUDA kernels: CPU reference derived from the .cu source vs Pascal on pascal-stack"
"$B/k3_ref" > "$B/k3_ref.txt"; "$B/test_k3" > "$B/k3_pas.txt"
"$B/compare" "$B/k3_ref.txt" "$B/k3_pas.txt" 0 1e-5
"$B/bfly_attn_ref" > "$B/bfly_ref.txt"; "$B/test_bfly_attn" > "$B/bfly_pas.txt"
"$B/compare" "$B/bfly_ref.txt" "$B/bfly_pas.txt" 0 1e-5
echo "== bit-exact view (tolerance 0) of the same CUDA comparisons"
"$B/compare" "$B/k3_ref.txt" "$B/k3_pas.txt" 0 0 | tail -1 || true
"$B/compare" "$B/bfly_ref.txt" "$B/bfly_pas.txt" 0 0 | tail -1 || true
echo "== self-checks (no runnable original)"
"$B/test_selfchecks"
