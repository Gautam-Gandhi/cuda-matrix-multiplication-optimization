#!/usr/bin/env python3
"""Figures from results/combined/{runtime_all,metrics_all}.csv.
Run after collect_runtime.py / parse_ncu.py.  Needs: pandas matplotlib seaborn.
Usage: python3 analysis/plots.py [results_dir]   ->  results/plots/*.png"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def save(out, name):
    plt.tight_layout(); plt.savefig(out / name, dpi=150); plt.close()


def fig_square(rt, out):
    sq = rt[rt.category == "square"]
    for gpu, g in sq.groupby("gpu"):
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=g, x="size", y="gflops", hue="kernel_name", marker="o")
        plt.title(f"Kernel performance vs square size — {gpu}")
        plt.xlabel("N (square N×N)"); plt.ylabel("GFLOP/s")
        plt.legend(fontsize=8, ncol=2)
        save(out, f"square_{gpu}.png")


def fig_cross(rt, out):
    sq = rt[rt.category == "square"]
    if sq.gpu.nunique() < 2:
        return
    sub = sq[sq.kernel_name.isin(["cuBLAS", "Warptiling", "2D Blocktiling", "Naive"])]
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=sub, x="size", y="gflops", hue="kernel_name", style="gpu", marker="o")
    plt.title("Cross-GPU: RTX 3050 (Ampere) vs T4 (Turing)")
    plt.xlabel("N"); plt.ylabel("GFLOP/s")
    save(out, "cross_gpu.png")


def fig_bar4096(rt, out):
    sq = rt[(rt.category == "square") & (rt["size"] == 4096)]
    for gpu, g in sq.groupby("gpu"):
        plt.figure(figsize=(10, 7))
        sns.barplot(data=g.sort_values("gflops"), y="kernel_name", x="gflops", color="steelblue")
        plt.title(f"GFLOP/s at N=4096 — {gpu}"); plt.xlabel("GFLOP/s"); plt.ylabel("")
        save(out, f"bar4096_{gpu}.png")


def fig_finesweep(rt, out):
    fs = rt[(rt.category == "arbitrary") & (rt.kernel == 0)]
    fs = fs[(fs.M == fs.N) & (fs.N == fs.K)]
    for gpu, g in fs.groupby("gpu"):
        g = g.sort_values("size")
        if len(g) < 8:
            continue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=g, x="size", y="gflops", marker="o")
        plt.title(f"cuBLAS tile-quantization sawtooth — {gpu}")
        plt.xlabel("N"); plt.ylabel("GFLOP/s")
        save(out, f"finesweep_{gpu}.png")


def fig_metrics(me, out):
    if me.empty:
        return
    show = ["dram_pct", "l2_pct", "l1_pct", "achieved_occupancy_pct",
            "registers_per_thread", "sectors_per_req_ld", "bank_conflicts_ld"]
    m = me[me.metric.isin(show)].copy()
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    if m.empty:
        return
    for gpu, g in m.groupby("gpu"):
        available = [s for s in show if s in g.metric.unique()]
        ncols = 2
        nrows = -(-len(available) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for ax_idx, metric in enumerate(available):
            sub = g[g.metric == metric].sort_values("kernel")
            axes_flat[ax_idx].bar(sub.kernel.astype(str), sub.value, color="steelblue")
            axes_flat[ax_idx].set_title(metric, fontsize=10)
            axes_flat[ax_idx].set_xlabel("Kernel")
            axes_flat[ax_idx].tick_params(axis="x", labelsize=8)
        for ax_idx in range(len(available), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)
        plt.suptitle(f"HW metrics by kernel — {gpu}", fontsize=13)
        save(out, f"metrics_{gpu}.png")


def fig_nonsquare(rt, out):
    ns = rt[rt.category == "nonsquare"].copy()
    if ns.empty:
        return
    ns["shape"] = ns.apply(lambda r: f"{int(r.M)}x{int(r.N)}x{int(r.K)}", axis=1)
    want = ["cuBLAS", "Warptiling", "2D Blocktiling", "Autotuned", "Naive"]
    sub = ns[ns.kernel_name.isin(want)]
    for gpu, g in sub.groupby("gpu"):
        plt.figure(figsize=(13, 6))
        sns.barplot(data=g, x="shape", y="gflops", hue="kernel_name",
                    hue_order=[k for k in want if k in g.kernel_name.unique()])
        plt.title(f"Non-square matrix performance — {gpu}")
        plt.xlabel("Shape (MxNxK)"); plt.ylabel("GFLOP/s")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.legend(fontsize=8)
        save(out, f"nonsquare_{gpu}.png")


def fig_paramsweep(root, out):
    for f in sorted(root.glob("*/paramsweep/kernel9.csv")):
        gpu = f.relative_to(root).parts[0]
        d = pd.read_csv(f)
        d["gflops"] = pd.to_numeric(d["gflops"], errors="coerce")
        piv = d.pivot_table(index="TM", columns="BK", values="gflops")
        annot_str = piv.map(lambda v: f"{v:.0f}" if pd.notna(v) else "x")
        plt.figure(figsize=(8, 6))
        sns.heatmap(piv, annot=annot_str, fmt="", cmap="viridis",
                    linewidths=0.5, linecolor="grey")
        plt.title(f"Kernel 9 GFLOP/s: BK (cols) vs TM=TN (rows) — {gpu}\n(x = build/launch failure)")
        plt.xlabel("BK"); plt.ylabel("TM = TN")
        save(out, f"paramsweep_k9_{gpu}.png")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = root / "plots"; out.mkdir(parents=True, exist_ok=True)
    rt = pd.read_csv(root / "combined" / "runtime_all.csv")
    # ensure numeric types and add convenience 'size' = N column
    for col in ("M", "N", "K", "gflops", "time_s", "kernel"):
        if col in rt.columns:
            rt[col] = pd.to_numeric(rt[col], errors="coerce")
    rt["size"] = rt["N"]
    mp = root / "combined" / "metrics_all.csv"
    me = pd.read_csv(mp) if mp.exists() else pd.DataFrame()
    print("Generating figures...")
    fig_square(rt, out);    print("  square done")
    fig_cross(rt, out);     print("  cross_gpu done")
    fig_bar4096(rt, out);   print("  bar4096 done")
    fig_finesweep(rt, out); print("  finesweep done")
    fig_metrics(me, out);   print("  metrics done")
    fig_nonsquare(rt, out); print("  nonsquare done")
    fig_paramsweep(root, out); print("  paramsweep done")
    print(f"\nAll figures -> {out}")


if __name__ == "__main__":
    main()
