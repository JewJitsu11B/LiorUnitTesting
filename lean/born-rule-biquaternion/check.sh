#!/usr/bin/env bash
# Lock-free single-file type-check against the prebuilt oleans (Carrier, Basic, Mathlib).
# Usage:  bash check.sh BornRule/SomeModule.lean
# Exit 0 = type-checks clean. Prints Lean errors/warnings otherwise.
# Safe to run concurrently (no lake build lock).
set -u
cd "$(dirname "$0")"
export PATH="$HOME/.elan/bin:$PATH"
LP=".lake/build/lib"
for p in Cli batteries Qq aesop proofwidgets importGraph LeanSearchClient plausible mathlib; do
  LP="$LP:.lake/packages/$p/.lake/build/lib"
done
export LEAN_PATH="$LP"
lean -DautoImplicit=false "$1"
