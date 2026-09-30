#!/usr/bin/env bash
# E4: Nsight Compute HW-metric profiling of selected kernels at one size.
# One representative launch per kernel (--set full, -c 1, filtered by name).
#
# GeForce (RTX 3050) counters need elevation AND the one-time driver unlock
# (scripts/enable_gpu_counters.reg + reboot). Run with:   SUDO=1 bash scripts/profile.sh
# On the Tesla T4 (Colab, root, no restriction) just:      bash scripts/profile.sh
# Output: results/$GPU/ncu/<gpu>_k<k>_s<size>.ncu-rep  (+ .csv metric dump)
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-rtx3050}"
DEVICE="${DEVICE:-0}"; export DEVICE
SIZE="${SIZE:-4096}"
KERNELS="${KERNELS:-1 2 3 4 5 6 9 10}"
O="results/$GPU/ncu"; mkdir -p "$O"

# Resolve ncu to an ABSOLUTE path: sudo resets PATH (secure_path), so a bare
# "sudo ncu" fails with "command not found". Allow NCU=/path override.
NCU="${NCU:-$(command -v ncu || echo /opt/nvidia/nsight-compute/2025.1.1/ncu)}"
[ -x "$NCU" ] || { echo "ncu not found at '$NCU' (set NCU=/path/to/ncu)"; exit 1; }
# SUDO=1 -> elevate (GeForce); default off (datacenter cards / Colab root).
PRE=""; [ "${SUDO:-0}" = "1" ] && PRE="sudo"

declare -A NAME=(
  [1]=sgemm_naive [2]=sgemm_global_mem_coalesce [3]=sgemm_shared_mem_block
  [4]=sgemm1DBlocktiling [5]=sgemm2DBlocktiling [6]=sgemmVectorize
  [7]=sgemmResolveBankConflicts [8]=sgemmResolveBankExtraCol
  [9]=sgemmAutotuned [10]=sgemmWarptiling)

for k in $KERNELS; do
  n="${NAME[$k]:-}"; [ -z "$n" ] && { echo "skip kernel $k (no name map)"; continue; }
  rep="$O/${GPU}_k${k}_s${SIZE}"
  echo "== ncu kernel $k ($n) @ $SIZE =="
  $PRE env SIZES="$SIZE" DEVICE="$DEVICE" "$NCU" --set full -k "regex:$n" -c 1 -f \
        --export "$rep" ./build/sgemm "$k" \
    || { echo "  ncu failed for kernel $k"; continue; }
  # dump every metric to CSV so downstream analysis needs no ncu installed
  $PRE "$NCU" -i "$rep.ncu-rep" --csv --page raw > "$rep.csv" 2>/dev/null || true
done
echo "DONE -> $O"
