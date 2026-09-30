#!/usr/bin/env bash
# E5: parameter-sensitivity sweep for kernel 9 (autotuned SGEMM) at size 4096.
# Sweeps BK x (TM=TN) holding BM=BN=128, NUM_THREADS=256. Mutates src/runner.cu
# in place and restores it afterwards. Illegal / too-large combos fail to build
# or launch on purpose -- that failure boundary IS the performance cliff.
# Output: results/$GPU/paramsweep/kernel9.csv
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-rtx3050}"; DEVICE="${DEVICE:-0}"; export DEVICE
O="results/$GPU/paramsweep"; mkdir -p "$O" temp
CSV="$O/kernel9.csv"; echo "kernel,BM,BN,BK,TM,TN,gflops" > "$CSV"

RUN=src/runner.cu
cp "$RUN" temp/runner.cu.bak
restore(){ cp temp/runner.cu.bak "$RUN"; }
trap restore EXIT
setc(){ sed -i "s/const uint $1 = [0-9]\+;/const uint $1 = $2;/" "$RUN"; }

for BK in 8 16 32 64; do
  for T in 2 4 8; do
    setc K9_BM 128; setc K9_BN 128; setc K9_BK "$BK"; setc K9_TM "$T"; setc K9_TN "$T"
    if make build >/dev/null 2>&1; then
      g=$(SIZES=4096 ./build/sgemm 9 2>/dev/null \
            | grep -oP 'performance: \(\s*\K[0-9.]+' | tail -1)
      echo "9,128,128,$BK,$T,$T,${g:-LAUNCH_FAIL}"
    else
      echo "9,128,128,$BK,$T,$T,BUILD_FAIL"
    fi | tee -a "$CSV"
  done
done

trap - EXIT
restore
make build >/dev/null 2>&1 || true
echo "DONE -> $CSV (src/runner.cu restored)"
