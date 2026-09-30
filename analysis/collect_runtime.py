#!/usr/bin/env python3
"""Parse results/*/runtime*/*.txt -> results/combined/runtime_all.csv.
Standard library only (no pandas), so it runs anywhere.
Usage: python3 analysis/collect_runtime.py [results_dir]"""
import csv, re, sys
from pathlib import Path

KERNEL_NAMES = {
    0: "cuBLAS", 1: "Naive", 2: "GMEM Coalescing", 3: "SMEM Caching",
    4: "1D Blocktiling", 5: "2D Blocktiling", 6: "Vectorized",
    7: "Bank Conflicts (Linearize)", 8: "Bank Conflicts (Offset)",
    9: "Autotuned", 10: "Warptiling", 11: "Double Buffering",
    12: "Double Buffering 2", 13: "cuBLAS TF32", 14: "cuBLAS BF16",
    15: "cuBLAS FP16",
}
CAT = {"runtime": "square", "runtime_nonsquare": "nonsquare",
       "runtime_arbitrary": "arbitrary"}
PERF = re.compile(
    r"performance: \(\s*([0-9.]+)\) GFLOPS\. size: \(\s*(\d+)\)\."
    r"(?: M=(\d+) N=(\d+) K=(\d+))?")
TIME = re.compile(r"elapsed time: \(\s*([0-9.]+)\)")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for f in sorted(root.glob("*/runtime*/*.txt")):
        gpu = f.relative_to(root).parts[0]
        cat = CAT.get(f.parent.name, f.parent.name)
        try:
            kernel = int(f.stem.split("_")[0])
        except ValueError:
            continue
        for line in f.read_text().splitlines():
            m = PERF.search(line)
            if not m:
                continue
            t = TIME.search(line)
            size = int(m.group(2))
            M = int(m.group(3)) if m.group(3) else size
            N = int(m.group(4)) if m.group(4) else size
            K = int(m.group(5)) if m.group(5) else size
            rows.append([gpu, cat, kernel, KERNEL_NAMES.get(kernel, str(kernel)),
                         M, N, K, float(m.group(1)),
                         float(t.group(1)) if t else ""])
    out = root / "combined"; out.mkdir(exist_ok=True)
    p = out / "runtime_all.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gpu", "category", "kernel", "kernel_name",
                    "M", "N", "K", "gflops", "time_s"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {p}")


if __name__ == "__main__":
    main()
