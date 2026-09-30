#!/usr/bin/env bash
# Runtime experiments: E1 square, E2 non-square, E3 arbitrary + fine sweep.
# Usage:  GPU=rtx3050 DEVICE=0 bash scripts/run_all.sh
# Output: results/$GPU/runtime*/<kernel>_output.txt
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-rtx3050}"
DEVICE="${DEVICE:-0}"; export DEVICE
BIN=./build/sgemm

make build >/dev/null || { echo "build failed"; exit 1; }

O="results/$GPU"
mkdir -p "$O/runtime" "$O/runtime_nonsquare" "$O/runtime_arbitrary"

# Ladder (0-10) + tensor-core cuBLAS (13 TF32, 14 BF16, 15 FP16).
# K11/K12 skipped (tuned for A6000-class shared memory).
# On the Tesla T4 override with: KERNELS="0 1 2 3 4 5 6 7 8 9 10 15"
KERNELS="${KERNELS:-0 1 2 3 4 5 6 7 8 9 10 13 14 15}"

echo "== E1 square sweep =="
for k in $KERNELS; do
  echo "  [square] kernel $k"
  $BIN "$k" | tee "$O/runtime/${k}_output.txt"
done

echo "== E2 non-square, tile-aligned (all kernels) =="
export MNK="4096,4096,4096;2048,8192,4096;8192,2048,4096;4096,4096,512;4096,4096,1024;4096,4096,2048"
for k in $KERNELS; do
  echo "  [nonsq] kernel $k"
  $BIN "$k" | tee "$O/runtime_nonsquare/${k}_output.txt"
done
unset MNK

echo "== E3 arbitrary / non-power-of-two (cuBLAS + naive + coalesce) =="
export MNK="3000,1500,3000;3000,3000,3000;1500,3000,1500;1000,1000,1000;1536,768,1536"
for k in 0 1 2; do
  echo "  [arb] kernel $k"
  $BIN "$k" | tee "$O/runtime_arbitrary/${k}_output.txt"
done
unset MNK

echo "== E3b cuBLAS fine square sweep (tile-quantization sawtooth) =="
SIZES="$(seq -s, 256 128 4096)" $BIN 0 | tee "$O/runtime_arbitrary/0_finesweep.txt"

echo "DONE -> $O"
