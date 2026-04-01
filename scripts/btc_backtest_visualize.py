from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path("result")
    high_file = root / "btc_scan_high" / "strategy_scan_20260401_104456.json"
    low_file = root / "btc_scan_low" / "strategy_scan_20260401_104327.json"
    out_dir = root / "btc_backtest_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(high_file, "r", encoding="utf-8") as f:
        high = json.load(f)
    with open(low_file, "r", encoding="utf-8") as f:
        low = json.load(f)

    high_rank = high["rankings"]["per_frequency_rankings"]["high"]
    low_rank = low["rankings"]["per_frequency_rankings"]["low"]

    for name, rank in [("high", high_rank), ("low", low_rank)]:
        top = rank[:10]
        strategies = [r["strategy"] for r in top][::-1]
        scores = [r["avg_score_across_assets"] for r in top][::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(strategies, scores)
        plt.xlabel("Avg CRPS (lower is better)")
        plt.title(f"BTC {name.upper()} Top-10 Strategies")
        plt.tight_layout()
        plt.savefig(out_dir / f"btc_{name}_top10_crps.png", dpi=160)
        plt.close()

    high_map = {r["strategy"]: r["avg_score_across_assets"] for r in high_rank}
    low_map = {r["strategy"]: r["avg_score_across_assets"] for r in low_rank}
    common = sorted(set(high_map).intersection(low_map), key=lambda s: high_map[s])

    plt.figure(figsize=(9, 9))
    for s in common:
        x, y = high_map[s], low_map[s]
        plt.scatter(x, y, s=28)
        if s in {"garch_v4", "regime_switching", "garch_v2", "dynamic_router", "production_baseline"}:
            plt.annotate(s, (x, y), fontsize=8)
    plt.xlabel("High Avg CRPS")
    plt.ylabel("Low Avg CRPS")
    plt.title("BTC Strategy Comparison: High vs Low")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "btc_high_vs_low_scatter.png", dpi=160)
    plt.close()

    summary_md = out_dir / "btc_backtest_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# BTC Backtest Summary\n\n")
        f.write(
            f"- High best: **{high['rankings']['per_asset_best']['BTC_high']['best_strategy']}** | "
            f"CRPS={high['rankings']['per_asset_best']['BTC_high']['avg_score']:.4f}\n"
        )
        f.write(
            f"- Low best: **{low['rankings']['per_asset_best']['BTC_low']['best_strategy']}** | "
            f"CRPS={low['rankings']['per_asset_best']['BTC_low']['avg_score']:.4f}\n\n"
        )
        f.write("## Top-5 HIGH\n")
        for i, r in enumerate(high_rank[:5], 1):
            f.write(f"{i}. {r['strategy']} — {r['avg_score_across_assets']:.4f}\n")
        f.write("\n## Top-5 LOW\n")
        for i, r in enumerate(low_rank[:5], 1):
            f.write(f"{i}. {r['strategy']} — {r['avg_score_across_assets']:.4f}\n")

    print(f"Saved visuals to {out_dir}")


if __name__ == "__main__":
    main()
