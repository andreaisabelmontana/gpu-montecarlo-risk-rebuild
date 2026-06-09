"""benchmark.py — run every available backend across a range of sizes,
verify VaR consistency, and emit a comparison table + chart.

It auto-detects which binaries exist (mc_serial, mc_openmp, mc_cuda) and which
runtimes are available (python), so it does the right thing on a CPU-only box
*and* on a CUDA machine. Outputs:
    results/benchmark.json   raw records
    results/benchmark.md     markdown table
    docs/throughput.png      sims/sec bar chart (log scale)

Usage:
    python scripts/benchmark.py [--sizes 100000,1000000,5000000] [--exe-dir .]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _exe(exe_dir: Path, name: str) -> Path | None:
    for cand in (exe_dir / name, exe_dir / f"{name}.exe"):
        if cand.exists():
            return cand
    return None


def run_backend(cmd: list[str]) -> dict | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        return None
    line = out.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,1000000,5000000")
    ap.add_argument("--exe-dir", default=str(ROOT))
    ap.add_argument("--seed", default="12345")
    ap.add_argument("--alpha", default="0.95")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    exe_dir = Path(args.exe_dir)

    # Discover backends.
    backends: list[tuple[str, list[str]]] = []
    py = shutil.which("python") or sys.executable
    backends.append(("python-numpy", [py, str(ROOT / "python" / "reference.py")]))
    for name, label in (("mc_serial", "serial"), ("mc_openmp", "openmp"), ("mc_cuda", "cuda")):
        exe = _exe(exe_dir, name)
        if exe:
            backends.append((label, [str(exe)]))

    print(f"Backends found: {[b[0] for b in backends]}")
    records = []
    for n in sizes:
        print(f"\n=== n_sims = {n:,} ===")
        for label, base in backends:
            # Keep the slow Python baseline to modest sizes.
            if label == "python-numpy" and n > 2_000_000:
                continue
            rec = run_backend(base + [str(n), args.seed, args.alpha])
            if rec:
                records.append(rec)
                print(f"  {label:>12}: {rec['sims_per_sec']:>14,.0f} sims/s  "
                      f"VaR={rec['VaR']:.2f}  CVaR={rec['CVaR']:.2f}  ({rec['seconds']:.3f}s)")

    # Persist raw records.
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "benchmark.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")

    # Markdown table at the largest common size.
    write_markdown(records)
    try:
        write_chart(records)
    except Exception as e:  # matplotlib optional
        print(f"(chart skipped: {e})")
    return 0


def write_markdown(records: list[dict]) -> None:
    if not records:
        return
    biggest = max(r["n_sims"] for r in records)
    rows = [r for r in records if r["n_sims"] == biggest]
    base = min((r["sims_per_sec"] for r in rows), default=1) or 1
    lines = [
        f"# Benchmark — {biggest:,} simulations\n",
        "| Backend | Threads | sims/sec | Speedup | VaR(95%) | CVaR(95%) | Time (s) |",
        "|---------|--------:|---------:|--------:|---------:|----------:|---------:|",
    ]
    for r in sorted(rows, key=lambda r: r["sims_per_sec"]):
        lines.append(
            f"| {r['backend']} | {r['threads']} | {r['sims_per_sec']:,.0f} | "
            f"{r['sims_per_sec'] / base:.1f}× | {r['VaR']:.2f} | {r['CVaR']:.2f} | "
            f"{r['seconds']:.3f} |"
        )
    (ROOT / "results" / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote results/benchmark.md")


def write_chart(records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        return
    biggest = max(r["n_sims"] for r in records)
    rows = sorted((r for r in records if r["n_sims"] == biggest),
                  key=lambda r: r["sims_per_sec"])
    labels = [r["backend"] for r in rows]
    vals = [r["sims_per_sec"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, vals, color="#38bdf8")
    ax.set_xscale("log")
    ax.set_xlabel("simulations / second (log scale)")
    ax.set_title(f"Monte Carlo risk engine — throughput @ {biggest:,} sims")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,.0f}", va="center", fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs" / "throughput.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
