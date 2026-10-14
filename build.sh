#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Builds every Pascal unit/program with fpc 3.2.2 against pascal-stack (read-only, never modified).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
PS=${PASCAL_STACK:-/workspace/pascal-stack}/handcrafted
mkdir -p "$ROOT/build"; cd "$ROOT/build"
FPC="fpc -Mobjfpc -Rintel -O2 -Sa -Fu$ROOT/src -Fu$ROOT/tests -Fi$ROOT/src -Fu$PS/core -Fu$PS/execution -FU$ROOT/build"
for p in src/k3_nano tests/test_k3 tests/test_bfly_attn tests/parity_rust_pas tests/parity_cpp_pas tests/parity_ffi_pas tests/test_selfchecks; do
  $FPC -o"$ROOT/build/$(basename $p)" "$ROOT/$p.pas"
done
