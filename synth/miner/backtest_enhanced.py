"""
backtest_enhanced.py

Backtest nâng cao: so sánh nhiều model, CRPS scoring, tuning, và visualization.
Lưu kết quả + charts vào result/

Usage:
    cd /Users/taiphan/Documents/synth
    conda activate synth
    python -m synth.miner.backtest_enhanced
"""

import json
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# ── Project imports ──────────────────────────────────────────────────
from synth.miner.my_simulation import fetch_price_data, simulate_crypto_price_paths
from synth.miner.core.garch_simulator import simulate_single_price_path_with_garch as garch_v1
from synth.miner.core.grach_simulator_v2 import simulate_single_price_path_with_garch as garch_v2
from synth.miner.core.HAR_RV_simulatior import simulate_single_price_path_with_har_garch as har_rv
from synth.miner.core.aparch_simulator import simulate_aparch_optimized as aparch

# ── Config ───────────────────────────────────────────────────────────
ASSETS = ["BTC", "ETH", "SOL"]
MODELS = {
    "GARCH_v1": garch_v1,
    "GARCH_v2": garch_v2,
    "HAR_RV":   har_rv,
    "APARCH":   aparch,
}

NUM_TEST_DATES = 5
DAYS_BACK_START = 10
DAYS_BACK_END = 3
TIME_INCREMENT = 300
TIME_LENGTH = 3600
NUM_SIMULATIONS = 100
SEED = 42

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result")
CHARTS_DIR = os.path.join(RESULT_DIR, "charts")
LOG_DIR = os.path.join(RESULT_DIR, "logs")


# ── Utility ──────────────────────────────────────────────────────────
class TeeLogger:
    def __init__(self, log_path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()


def generate_test_dates(num_dates, days_back_start, days_back_end, seed=42):
    import random
    random.seed(seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back_start)
    end = now - timedelta(days=days_back_end)
    dates = []
    for _ in range(num_dates):
        delta = (end - start).total_seconds()
        dt = start + timedelta(seconds=random.uniform(0, delta))
        dates.append(dt.replace(second=0, microsecond=0))
    dates.sort()
    return dates


def compute_crps_simple(predictions: np.ndarray, real_prices: list) -> float:
    """
    Simplified CRPS calculation for model comparison.
    predictions: shape (n_sims, steps+1)
    real_prices: list of actual prices at each time step
    """
    if predictions is None or len(real_prices) == 0:
        return float("nan")

    n_steps = min(predictions.shape[1], len(real_prices))
    crps_values = []

    for t in range(n_steps):
        obs = real_prices[t]
        ensemble = np.sort(predictions[:, t])
        n = len(ensemble)

        # CRPS = E|X - y| - 0.5 * E|X - X'|
        mae = np.mean(np.abs(ensemble - obs))
        # Efficient pairwise mean absolute difference
        weights = 2 * np.arange(1, n + 1) - n - 1
        spread = np.sum(weights * ensemble) / (n * n)
        crps_values.append(mae - spread)

    return float(np.mean(crps_values))


def iso_to_timestamp(iso_str):
    dt = datetime.fromisoformat(iso_str)
    return int(dt.timestamp())


# ── Run single model test ────────────────────────────────────────────
def run_model_test(asset, start_time, model_name, model_fn, time_increment, time_length, num_sims, seed):
    """Run a single simulation with a specific model and return paths + metadata."""
    t0 = time.time()
    try:
        paths = simulate_crypto_price_paths(
            current_price=None,
            asset=asset,
            start_time=start_time.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_sims,
            simulate_fn=model_fn,
            max_data_points=None,
            seed=seed,
        )
        elapsed = time.time() - t0

        if paths is None or len(paths) == 0:
            return {"status": "FAIL", "error": "No paths generated", "elapsed": elapsed}

        return {
            "status": "SUCCESS",
            "paths": paths,
            "num_paths": paths.shape[0],
            "path_length": paths.shape[1],
            "price_start": float(paths[0, 0]),
            "price_end": float(np.mean(paths[:, -1])),
            "price_std_end": float(np.std(paths[:, -1])),
            "elapsed": round(elapsed, 3),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    [ERROR] {model_name}: {e}")
        traceback.print_exc()
        return {"status": "ERROR", "error": str(e), "elapsed": round(elapsed, 3)}


# ── Visualization ────────────────────────────────────────────────────
def plot_simulation_paths(results_by_model, asset, date_str, save_dir):
    """Plot simulation paths for all models side by side."""
    models_ok = {k: v for k, v in results_by_model.items() if v["status"] == "SUCCESS"}
    if not models_ok:
        return

    n_models = len(models_ok)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = {"GARCH_v1": "#e74c3c", "GARCH_v2": "#3498db", "HAR_RV": "#2ecc71", "APARCH": "#9b59b6"}

    for ax, (model_name, res) in zip(axes, models_ok.items()):
        paths = res["paths"]
        n_show = min(30, paths.shape[0])
        steps = np.arange(paths.shape[1]) * TIME_INCREMENT / 60  # minutes

        for i in range(n_show):
            ax.plot(steps, paths[i], alpha=0.15, color=colors.get(model_name, "#666"), linewidth=0.7)

        mean_path = np.mean(paths, axis=0)
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        ax.plot(steps, mean_path, color=colors.get(model_name, "#333"), linewidth=2, label="Mean")
        ax.fill_between(steps, p5, p95, alpha=0.15, color=colors.get(model_name, "#666"), label="90% CI")
        ax.set_title(f"{model_name}\n(σ_end={res['price_std_end']:.1f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Minutes")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Price ($)")
    fig.suptitle(f"{asset} — Simulation Paths Comparison\n{date_str}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, f"{asset}_{date_str[:10]}_paths.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison_bar(all_results, save_dir):
    """Bar chart comparing models across assets by time and success rate."""
    # Aggregate data
    summary = {}
    for r in all_results:
        key = (r["asset"], r["model"])
        if key not in summary:
            summary[key] = {"times": [], "successes": 0, "total": 0}
        summary[key]["total"] += 1
        if r["status"] == "SUCCESS":
            summary[key]["successes"] += 1
            summary[key]["times"].append(r["elapsed"])

    assets = sorted(set(r["asset"] for r in all_results))
    models = list(MODELS.keys())
    colors = {"GARCH_v1": "#e74c3c", "GARCH_v2": "#3498db", "HAR_RV": "#2ecc71", "APARCH": "#9b59b6"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Average execution time
    x = np.arange(len(assets))
    width = 0.18
    for i, model in enumerate(models):
        vals = []
        for asset in assets:
            key = (asset, model)
            if key in summary and summary[key]["times"]:
                vals.append(np.mean(summary[key]["times"]))
            else:
                vals.append(0)
        ax1.bar(x + i * width, vals, width, label=model, color=colors.get(model, "#999"), alpha=0.85)

    ax1.set_xlabel("Asset")
    ax1.set_ylabel("Avg Time (s)")
    ax1.set_title("Average Execution Time by Model", fontweight="bold")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(assets)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Chart 2: Success rate
    for i, model in enumerate(models):
        vals = []
        for asset in assets:
            key = (asset, model)
            if key in summary:
                rate = summary[key]["successes"] / max(summary[key]["total"], 1) * 100
                vals.append(rate)
            else:
                vals.append(0)
        ax2.bar(x + i * width, vals, width, label=model, color=colors.get(model, "#999"), alpha=0.85)

    ax2.set_xlabel("Asset")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Model Success Rate", fontweight="bold")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(assets)
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_volatility_comparison(all_results, save_dir):
    """Compare predicted volatility (std of end prices) across models."""
    assets = sorted(set(r["asset"] for r in all_results))
    models = list(MODELS.keys())
    colors = {"GARCH_v1": "#e74c3c", "GARCH_v2": "#3498db", "HAR_RV": "#2ecc71", "APARCH": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(assets))
    width = 0.18

    for i, model in enumerate(models):
        vals = []
        for asset in assets:
            stds = [r["price_std_end"] for r in all_results
                    if r["asset"] == asset and r["model"] == model and r["status"] == "SUCCESS"]
            vals.append(np.mean(stds) if stds else 0)
        ax.bar(x + i * width, vals, width, label=model, color=colors.get(model, "#999"), alpha=0.85)

    ax.set_xlabel("Asset")
    ax.set_ylabel("Avg σ (End Price Std Dev)")
    ax.set_title("Predicted Volatility by Model (Lower = Tighter Predictions)", fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(assets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "volatility_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_heatmap(all_results, save_dir):
    """Create a summary heatmap of GARCH parameters across assets."""
    # Collect parameter data from results
    param_data = {}
    for r in all_results:
        if r["status"] == "SUCCESS" and r["model"] == "GARCH_v2":
            asset = r["asset"]
            if asset not in param_data:
                param_data[asset] = {"elapsed": [], "std_end": [], "paths": []}
            param_data[asset]["elapsed"].append(r["elapsed"])
            param_data[asset]["std_end"].append(r["price_std_end"])
            param_data[asset]["paths"].append(r["num_paths"])

    if not param_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    assets = sorted(param_data.keys())
    metrics = ["Avg Time (s)", "Avg σ End", "Paths"]

    data = []
    for asset in assets:
        d = param_data[asset]
        data.append([
            np.mean(d["elapsed"]),
            np.mean(d["std_end"]),
            np.mean(d["paths"]),
        ])

    data = np.array(data)
    # Normalize each column for coloring
    norm_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

    im = ax.imshow(norm_data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(assets)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(assets)

    for i in range(len(assets)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_title("GARCH v2 Performance Heatmap", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = os.path.join(save_dir, "parameter_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ─────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"enhanced_backtest_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 80)
    print(f"ENHANCED BACKTEST — Multi-Model Comparison — {timestamp}")
    print(f"Assets: {ASSETS}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Test dates: {NUM_TEST_DATES}")
    print(f"Simulation: {TIME_LENGTH}s length, {TIME_INCREMENT}s inc, {NUM_SIMULATIONS} sims")
    print("=" * 80)

    dates = generate_test_dates(NUM_TEST_DATES, DAYS_BACK_START, DAYS_BACK_END, SEED)
    print(f"\nTest dates:")
    for i, d in enumerate(dates):
        print(f"  [{i+1}] {d.isoformat()}")

    all_results = []
    chart_paths = []

    for asset in ASSETS:
        print(f"\n{'='*60}")
        print(f"ASSET: {asset}")
        print(f"{'='*60}")

        # Pre-fetch data once for all models
        print(f"[INFO] Pre-fetching data for {asset}...")
        fetch_price_data(asset, TIME_INCREMENT, only_load=False)

        for i, dt in enumerate(dates):
            date_str = dt.strftime("%Y-%m-%d_%H%M")
            print(f"\n--- {asset} Date {i+1}/{len(dates)}: {dt.isoformat()} ---")

            model_results = {}
            for model_name, model_fn in MODELS.items():
                print(f"  Running {model_name}...")
                res = run_model_test(
                    asset, dt, model_name, model_fn,
                    TIME_INCREMENT, TIME_LENGTH, NUM_SIMULATIONS, SEED
                )

                result_entry = {
                    "asset": asset,
                    "date": dt.isoformat(),
                    "model": model_name,
                    "status": res["status"],
                    "elapsed": res.get("elapsed", 0),
                    "num_paths": res.get("num_paths", 0),
                    "path_length": res.get("path_length", 0),
                    "price_start": res.get("price_start"),
                    "price_end": res.get("price_end"),
                    "price_std_end": res.get("price_std_end", 0),
                    "error": res.get("error"),
                }
                all_results.append(result_entry)
                model_results[model_name] = res

                status = "✅" if res["status"] == "SUCCESS" else "❌"
                print(f"    {status} {model_name}: {res['status']} ({res.get('elapsed', 0)}s)")
                if res["status"] == "SUCCESS":
                    print(f"       paths={res['num_paths']}, σ_end={res.get('price_std_end', 0):.1f}")

            # Plot paths comparison for this date
            cp = plot_simulation_paths(model_results, asset, date_str, CHARTS_DIR)
            if cp:
                chart_paths.append(cp)

    # ── Generate summary charts ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATING CHARTS...")
    print(f"{'='*60}")

    cp1 = plot_model_comparison_bar(all_results, CHARTS_DIR)
    if cp1: chart_paths.append(cp1)

    cp2 = plot_volatility_comparison(all_results, CHARTS_DIR)
    if cp2: chart_paths.append(cp2)

    cp3 = plot_parameter_heatmap(all_results, CHARTS_DIR)
    if cp3: chart_paths.append(cp3)

    # ── Save results ─────────────────────────────────────────────────
    full_path = os.path.join(RESULT_DIR, f"enhanced_backtest_{timestamp}.json")
    with open(full_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "assets": ASSETS,
                "models": list(MODELS.keys()),
                "num_test_dates": NUM_TEST_DATES,
                "time_increment": TIME_INCREMENT,
                "time_length": TIME_LENGTH,
                "num_simulations": NUM_SIMULATIONS,
            },
            "results": all_results,
            "charts": chart_paths,
        }, f, indent=2, default=str)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ENHANCED BACKTEST SUMMARY")
    print(f"{'='*80}")

    # Model × Asset summary table
    print(f"\n{'Model':<12} {'Asset':<6} {'Pass':>5} {'Fail':>5} {'AvgTime':>8} {'Avg σ_end':>10}")
    print("-" * 50)
    for model in MODELS:
        for asset in ASSETS:
            subset = [r for r in all_results if r["model"] == model and r["asset"] == asset]
            ok = [r for r in subset if r["status"] == "SUCCESS"]
            fail = len(subset) - len(ok)
            avg_t = np.mean([r["elapsed"] for r in ok]) if ok else 0
            avg_std = np.mean([r["price_std_end"] for r in ok]) if ok else 0
            icon = "✅" if fail == 0 else "⚠️"
            print(f"{icon} {model:<10} {asset:<6} {len(ok):>5} {fail:>5} {avg_t:>7.2f}s {avg_std:>9.1f}")

    total_ok = sum(1 for r in all_results if r["status"] == "SUCCESS")
    total = len(all_results)
    print(f"\nTOTAL: {total_ok}/{total} passed")

    # Model ranking by volatility (lower σ = tighter = better for CRPS)
    print(f"\n{'='*40}")
    print("MODEL RANKING (by avg σ_end — lower is better for CRPS)")
    print(f"{'='*40}")
    model_scores = {}
    for model in MODELS:
        stds = [r["price_std_end"] for r in all_results if r["model"] == model and r["status"] == "SUCCESS"]
        model_scores[model] = np.mean(stds) if stds else float("inf")
    for rank, (model, score) in enumerate(sorted(model_scores.items(), key=lambda x: x[1]), 1):
        print(f"  #{rank} {model}: avg σ = {score:.1f}")

    # Tuning recommendations
    print(f"\n{'='*40}")
    print("TUNING RECOMMENDATIONS")
    print(f"{'='*40}")
    best_model = min(model_scores, key=model_scores.get)
    print(f"  Best model: {best_model} (lowest volatility spread)")
    print(f"  → For CRPS optimization, tighter predictions score better")
    print(f"  → Consider increasing lookback_days for more stable fitting")
    print(f"  → For BTC: lookback_days=45 (5m) seems optimal per sn50")
    print(f"  → vol_multiplier tuning: try 0.95-1.05 range")

    print(f"\nFiles saved:")
    print(f"  Results: {full_path}")
    print(f"  Log: {log_path}")
    print(f"  Charts ({len(chart_paths)}):")
    for cp in chart_paths:
        print(f"    → {os.path.basename(cp)}")
    print("=" * 80)

    sys.stdout = logger.stdout
    logger.close()

    print(f"\n✅ Enhanced backtest complete: {total_ok}/{total} passed")
    print(f"   Results: {RESULT_DIR}")
    print(f"   Charts: {CHARTS_DIR} ({len(chart_paths)} files)")
    print(f"   Log: {log_path}")


if __name__ == "__main__":
    main()
