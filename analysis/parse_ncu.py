#!/usr/bin/env python3
"""Extract key metrics from results/*/ncu/*.csv (ncu `--csv --page raw` dumps)
into results/combined/metrics_all.csv. Standard library only.

The raw page is WIDE (one row per kernel, one column per metric) and prefixes
some metric ids with a section tag (e.g. "TriageSCG.sm__throughput..."), so we
match columns by exact name or dotted suffix. A few assignment-critical figures
(coalescing = sectors/request) aren't emitted directly, so we derive them.
Usage: python3 analysis/parse_ncu.py [results_dir]"""
import csv, re, sys
from pathlib import Path

# friendly label -> raw ncu metric id (matched as col == id or col endswith "."+id)
METRICS = {
    "compute_sol_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_sol_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_per_block_static": "launch__shared_mem_per_block_static",
    "smem_per_block_dynamic": "launch__shared_mem_per_block_dynamic",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "duration_ns": "gpu__time_duration.sum",
    "bank_conflicts_ld": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "bank_conflicts_st": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
    # building blocks for the derived coalescing ratios below
    "_g_ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "_g_ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "_g_st_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "_g_st_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
}
FN = re.compile(r"_k(\d+)_s(\d+)\.csv$")


def num(s):
    s = str(s).replace(",", "").strip().strip('"')
    try:
        return float(s)
    except ValueError:
        return None


# ncu auto-scales units per kernel (naive duration prints in 's', the rest in
# 'ms'; static smem in 'Kbyte' vs 'byte'; DRAM in 'Gbyte' vs 'Mbyte'), so the
# raw value column is NOT comparable across kernels. Fold every time -> ns and
# every byte -> byte so one metric always means one unit. The '/block','/req'
# etc. suffix is preserved.
_TIME = {"s": 1e9, "second": 1e9, "ms": 1e6, "msecond": 1e6,
         "us": 1e3, "usecond": 1e3, "ns": 1.0, "nsecond": 1.0}
_BYTE = {"Gbyte": 1e9, "Mbyte": 1e6, "Kbyte": 1e3, "byte": 1.0}


def normalize(value, unit):
    """(value, unit) -> (value_in_base, base_unit); identity for other units."""
    head, sep, tail = (unit or "").partition("/")
    for table, base in ((_TIME, "ns"), (_BYTE, "byte")):
        if head in table:
            return round(value * table[head], 4), base + (sep + tail)
    return value, unit


def get_val(cols, row, target):
    """Value for a metric id. Raw dumps repeat a metric under a section-prefixed
    column (often blank) and a bare column (the real value), so prefer the exact
    bare name and skip blanks."""
    cands = [c for c in cols if c == target or c.endswith("." + target)]
    cands.sort(key=lambda c: c != target)   # exact bare name first
    for c in cands:
        v = num(row.get(c))
        if v is not None:
            return v, c
    return None, None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for f in sorted(root.glob("*/ncu/*.csv")):
        m = FN.search(f.name)
        if not m:
            continue
        gpu = f.relative_to(root).parts[0]
        kernel, size = int(m.group(1)), int(m.group(2))
        with f.open(newline="") as fh:
            rd = csv.reader(fh)
            cols = next(rd, None)          # row 1: metric names
            units = next(rd, None)         # row 2: units (%, block, cycle, ...)
            data = next(rd, None)          # row 3: the (only) kernel's values; -c 1
        if not cols or not data:
            continue
        row = dict(zip(cols, data))
        unit = dict(zip(cols, units or []))
        vals = {}
        for label, mid in METRICS.items():
            v, col = get_val(cols, row, mid)
            if v is None:
                continue
            vals[label] = v            # raw count, for the derived ratios below
            if not label.startswith("_"):
                nv, nu = normalize(v, unit.get(col, ""))
                rows.append([gpu, kernel, size, label, mid, nv, nu])
        # derived coalescing: sectors per request (lower = better coalescing)
        for tag, s, r in (("sectors_per_req_ld", "_g_ld_sectors", "_g_ld_requests"),
                          ("sectors_per_req_st", "_g_st_sectors", "_g_st_requests")):
            if vals.get(r):
                rows.append([gpu, kernel, size, tag, "derived",
                             round(vals[s] / vals[r], 4), "sectors/req"])

    out = root / "combined"; out.mkdir(exist_ok=True)
    p = out / "metrics_all.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gpu", "kernel", "size", "metric", "ncu_name", "value", "unit"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {p}")


if __name__ == "__main__":
    main()
