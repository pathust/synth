"""
tune_garch_grid.py — Tournament tuning for GARCH-family strategies.

Concept:
  - Each **individual** is (strategy_name, params). Same strategy with different params
    counts as a different individual.
  - Backtest over a fixed UTC day range (default 2026-03-21 .. 2026-04-10 inclusive).
  - Every ``--window-days`` (default 4) days, evaluate all remaining individuals on that
    window and eliminate the worst ``--cut-ratio`` fraction by avg metric score
    (CRPS/RMSE/MAE lower is better; DIR_ACC higher is better).
  - Cadence (start_time only): **high=5m**, **low=12m** (same as compare_entries). Scoring
    remains as implemented by BacktestRunner (high mean per hour, low mean per day).

Output:
  - JSON report with per-window cuts and survivors
  - Draft YAML (winner per asset, copied under every regime for that asset type)
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Importing backtest.runner pulls bittensor, which handles global --help before our argparse.
_SHOW_HELP = False
if __name__ == "__main__" and (
    "--help" in sys.argv or "-h" in sys.argv
):
    _SHOW_HELP = True
    sys.argv = [a for a in sys.argv if a not in ("--help", "-h")]

import argparse

warnings.filterwarnings("ignore", category=Warning, module="arch")
warnings.filterwarnings("ignore", message=".*poorly scaled.*")
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*A date index has been provided.*")

import yaml

from synth.miner.backtest.runner import BacktestRunner
from synth.miner.strategies import StrategyRegistry
from synth.miner.strategies.base import REGIME_TYPES, get_asset_type


DEFAULT_STRATEGIES = [
    "garch_v1",
    "garch_v2_2",
    "garch_v2",
    "garch_v2_1",
    "garch_v4",
    "ensemble_garch_v2_v4",
]


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _enumerate_all_times_in_days(
    start_day: date,
    end_day: date,
    *,
    frequency: str,
) -> list[datetime]:
    """Start_times at cadence high=300s, low=720s within [start_day, end_day] inclusive."""
    if end_day < start_day:
        return []
    start_dt = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc)
    end_exclusive = (
        datetime(end_day.year, end_day.month, end_day.day, tzinfo=timezone.utc) + timedelta(days=1)
    )
    step_seconds = 300 if frequency == "high" else 720
    step = timedelta(seconds=step_seconds)
    out: list[datetime] = []
    cur = start_dt
    while cur < end_exclusive:
        out.append(cur)
        cur = cur + step
    return out


def _parse_day_utc(s: str) -> date:
    parts = str(s).strip().split("-")
    if len(parts) != 3:
        raise SystemExit(f"Invalid day '{s}', expected YYYY-MM-DD")
    y, m, d = (int(p) for p in parts)
    return date(y, m, d)


def _enumerate_all_times_in_range(
    start_day: date,
    end_day: date,
    *,
    frequency: str,
) -> list[datetime]:
    """Enumerate cadence slots for every day in [start_day, end_day] inclusive."""
    if end_day < start_day:
        return []
    out: list[datetime] = []
    cur = start_day
    while cur <= end_day:
        out.extend(_enumerate_all_times_in_days(cur, cur, frequency=frequency))
        cur = cur + timedelta(days=1)
    return out


def _param_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def _to_yaml_serializable(obj: Any) -> Any:
    """Recursively convert numpy scalars and nested structures for PyYAML."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return obj
    # Avoid importing numpy here (dev env may not have it); handle numpy scalars by duck-typing.
    if hasattr(obj, "item") and callable(getattr(obj, "item")) and obj.__class__.__module__.startswith("numpy"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _to_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_serializable(v) for v in obj]
    return obj


def _metric_prefers_lower(metric: str) -> bool:
    return metric.upper() != "DIR_ACC"


def pick_best_individual_per_asset(
    rows: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, dict[str, Any]]:
    """
    One winner per asset: best ``full_range_score`` among tuned individuals for that asset.
    CRPS/RMSE/MAE → lower is better; DIR_ACC → higher is better.
    """
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        a = str(r.get("asset", ""))
        if not a:
            continue
        by_asset.setdefault(a, []).append(r)
    lower = _metric_prefers_lower(metric)
    winners: dict[str, dict[str, Any]] = {}
    for asset, items in sorted(by_asset.items()):
        best = items[0]
        bs = float(best["avg_window_score"])
        for r in items[1:]:
            s = float(r["avg_window_score"])
            if lower:
                if s < bs:
                    best, bs = r, s
            else:
                if s > bs:
                    best, bs = r, s
        winners[asset] = best
    return winners


def build_draft_yaml_document(
    *,
    winners: dict[str, dict[str, Any]],
    frequency: str,
    metric: str,
    start_day: str,
    end_day: str,
    window_days: int,
    cut_ratio: float,
    seed: int,
    json_report: str,
) -> dict[str, Any]:
    """
    Draft routing: same top-1 model + params under every regime for each asset's type
    (crypto / gold / equity), for the tuned frequency only.
    """
    routing: dict[str, Any] = {}
    winners_summary: dict[str, Any] = {}
    for asset in sorted(winners.keys()):
        row = winners[asset]
        name = str(row["strategy"])
        params = _to_yaml_serializable(row.get("best_params") or {})
        cell = {
            "models": [
                {
                    "name": name,
                    "weight": 1.0,
                    "params": params,
                }
            ]
        }
        regimes = REGIME_TYPES.get(get_asset_type(asset), REGIME_TYPES["crypto"])
        routing[asset] = {frequency: {reg: cell for reg in regimes}}
        winners_summary[asset] = {
            "strategy": name,
            "avg_window_score": row.get("avg_window_score"),
        }

    return {
        "version": "2.2",
        "source": {
            "method": "tune_garch_grid",
            "metric": metric,
            "selection": "tournament_best_full_range_per_asset",
            "json_report": json_report,
        },
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tuning": {
            "seed": seed,
            "start_day": start_day,
            "end_day": end_day,
            "window_days": int(window_days),
            "cut_ratio": float(cut_ratio),
        },
        "winners_summary": winners_summary,
        "routing": routing,
    }


def _score_individual(
    runner: BacktestRunner,
    *,
    strategy_obj,
    strategy_name: str,
    asset: str,
    frequency: str,
    params: dict[str, Any],
    slots: list[datetime],
    num_sims: int,
    seed: int,
) -> float:
    res = runner.run_benchmark(
        strategy_obj,
        asset,
        frequency,
        num_runs=len(slots),
        num_sims=num_sims,
        seed=seed,
        window_days=1,
        dates=slots,
        **params,
    )
    return float(res["avg_score"])


def tournament_tune_asset(
    runner: BacktestRunner,
    registry: StrategyRegistry,
    *,
    asset: str,
    frequency: str,
    strategy_names: list[str],
    start_day: date,
    end_day: date,
    window_days: int,
    cut_ratio: float,
    num_sims: int,
    seed: int,
) -> dict[str, Any]:
    """
    Tournament elimination:
      - each individual is (strategy_name, params)
      - each window ranks by avg score and removes the worst cut_ratio fraction
      - if a cut would leave <= 3 survivors (or population already <= 3), stop cutting:
        keep everyone and only accumulate window scores until end_date
      - best is chosen by avg window score among final survivors (>=2 when possible)
    """
    # If the next cut would leave at most this many individuals, skip cutting and run out.
    _STOP_CUT_SURVIVOR_CAP = 5
    cadence_s = 300 if frequency == "high" else 720
    lower_is_better = _metric_prefers_lower(runner.metric)

    # Build population
    population: list[tuple[str, dict[str, Any]]] = []
    for name in strategy_names:
        try:
            s = registry.get(name)
        except KeyError:
            print(f"[skip] Unknown strategy: {name}")
            continue
        if (not s.supports_frequency(frequency)) or (not s.supports_asset(asset)):
            continue
        grid = s.get_param_grid(frequency=frequency, asset=asset)
        for params in _param_combos(grid):
            population.append((name, params))

    if not population:
        raise SystemExit(f"No compatible individuals for asset={asset} frequency={frequency}")

    print(
        f"\n[Tournament] {asset} {frequency}: cadence={cadence_s}s "
        f"population={len(population)} range={start_day}..{end_day} "
        f"window_days={window_days} cut_ratio={cut_ratio}"
    )

    rounds: list[dict[str, Any]] = []
    # Track per-individual window scores for survivors-only selection.
    # Key = (strategy_name, json-dumped params with sorted keys) to make it hashable/stable.
    score_hist: dict[tuple[str, str], list[float]] = {}
    def _key(ind: tuple[str, dict[str, Any]]) -> tuple[str, str]:
        n, p = ind
        return (str(n), json.dumps(p, sort_keys=True, default=str))

    cur_start = start_day
    round_idx = 0
    freeze_elimination = False
    while cur_start <= end_day:
        cur_end = min(end_day, cur_start + timedelta(days=int(window_days) - 1))
        slots = _enumerate_all_times_in_range(cur_start, cur_end, frequency=frequency)
        if not slots:
            break

        scored: list[tuple[float, tuple[str, dict[str, Any]]]] = []
        for strat_name, params in population:
            strat_obj = registry.get(strat_name)
            score = _score_individual(
                runner,
                strategy_obj=strat_obj,
                strategy_name=strat_name,
                asset=asset,
                frequency=frequency,
                params=params,
                slots=slots,
                num_sims=num_sims,
                seed=seed + 10000 * round_idx,
            )
            score_hist.setdefault(_key((strat_name, params)), []).append(float(score))
            scored.append((score, (strat_name, params)))

        scored.sort(key=lambda t: t[0], reverse=(not lower_is_better))
        n_before = len(scored)
        if freeze_elimination or n_before <= _STOP_CUT_SURVIVOR_CAP:
            n_keep = n_before
            elimination = "frozen"
        else:
            n_keep = max(1, int((1.0 - float(cut_ratio)) * n_before + 0.00001))
            if n_keep <= _STOP_CUT_SURVIVOR_CAP:
                freeze_elimination = True
                n_keep = n_before
                elimination = "frozen_would_leave_too_few"
            else:
                elimination = "cut"
        kept = scored[:n_keep]

        rounds.append(
            {
                "round": round_idx + 1,
                "window": {"start": cur_start.isoformat(), "end": cur_end.isoformat()},
                "slots": len(slots),
                "population_before": n_before,
                "population_after": n_keep,
                "elimination": elimination,
                "best_score": float(kept[0][0]),
                "cut_score_threshold": float(kept[-1][0]),
            }
        )
        print(
            f"  round {round_idx + 1}: {cur_start}..{cur_end} slots={len(slots)} "
            f"pop {n_before}->{n_keep} ({elimination}) best={kept[0][0]:.6f}"
        )

        population = [ind for _, ind in kept]
        cur_start = cur_end + timedelta(days=1)
        round_idx += 1

    # If range is shorter than one window, we still need at least one scoring pass.
    if not rounds:
        slots = _enumerate_all_times_in_range(start_day, end_day, frequency=frequency)
        scored: list[tuple[float, tuple[str, dict[str, Any]]]] = []
        for strat_name, params in population:
            strat_obj = registry.get(strat_name)
            score = _score_individual(
                runner,
                strategy_obj=strat_obj,
                strategy_name=strat_name,
                asset=asset,
                frequency=frequency,
                params=params,
                slots=slots,
                num_sims=num_sims,
                seed=seed,
            )
            score_hist.setdefault(_key((strat_name, params)), []).append(float(score))
            scored.append((score, (strat_name, params)))
        scored.sort(key=lambda t: t[0], reverse=(not lower_is_better))
        rounds.append(
            {
                "round": 1,
                "window": {"start": start_day.isoformat(), "end": end_day.isoformat()},
                "slots": len(slots),
                "population_before": len(scored),
                "population_after": len(scored),
                "best_score": float(scored[0][0]),
                "cut_score_threshold": float(scored[-1][0]),
            }
        )

    # Final selection among survivors that made it to the end:
    # best = min/ max over avg(window_scores) across all windows in the range.
    survivors: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for strat_name, params in population:
        hist = score_hist.get(_key((strat_name, params)), [])
        if not hist:
            # Should not happen, but guard.
            avg_win = float("inf") if lower_is_better else float("-inf")
        else:
            avg_win = float(sum(hist) / len(hist))
        row = {
            "asset": asset,
            "frequency": frequency,
            "strategy": strat_name,
            "best_params": params,
            "avg_window_score": float(avg_win),
            "window_scores": hist,
        }
        survivors.append(row)
        if best is None:
            best = row
        else:
            if lower_is_better:
                if row["avg_window_score"] < best["avg_window_score"]:
                    best = row
            else:
                if row["avg_window_score"] > best["avg_window_score"]:
                    best = row

    if best is None:
        raise SystemExit("No survivor after tournament")

    print(
        f"[Tournament] best {asset} {frequency}: {best['strategy']} "
        f"avg_window_{runner.metric}={best['avg_window_score']:.6f}"
    )
    return {
        "asset": asset,
        "frequency": frequency,
        "request_cadence_s": cadence_s,
        "rounds": rounds,
        "survivors": survivors,
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tournament tuning: each (strategy, params) is an individual; "
            "every window eliminates the worst fraction by avg metric score; "
            "outputs winners + a draft YAML."
        )
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH", "SOL", "XAU"],
        help="Assets to tune",
    )
    parser.add_argument(
        "--frequency",
        choices=["high", "low"],
        default="high",
        help="high=5m start_time cadence (1m data); low=12m cadence (5m data)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="Strategy registry names",
    )
    parser.add_argument("--num-sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-day",
        default="2026-03-21",
        help="UTC start day (YYYY-MM-DD), inclusive",
    )
    parser.add_argument(
        "--end-day",
        default="2026-04-10",
        help="UTC end day (YYYY-MM-DD), inclusive",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=4,
        help="Evaluate + cut every N calendar days",
    )
    parser.add_argument(
        "--cut-ratio",
        type=float,
        default=0.5,
        help="Eliminate the worst fraction per window (0..1)",
    )
    parser.add_argument(
        "--metric",
        default="CRPS",
        choices=["CRPS", "RMSE", "MAE", "DIR_ACC"],
    )
    parser.add_argument(
        "--result-dir",
        default="result/tune_garch_grid",
        help="Output directory for JSON report",
    )
    parser.add_argument(
        "--draft-yaml",
        default="",
        metavar="PATH",
        help=(
            "Write strategies draft YAML (best test metric per asset). "
            "Default: <result-dir>/strategies_draft_garch_grid_<timestamp>.yaml"
        ),
    )
    parser.add_argument(
        "--no-draft-yaml",
        action="store_true",
        help="Do not write the draft YAML file",
    )
    if _SHOW_HELP:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args()
    cadence_s = 300 if args.frequency == "high" else 720
    start_day = _parse_day_utc(str(args.start_day))
    end_day = _parse_day_utc(str(args.end_day))
    if end_day < start_day:
        raise SystemExit("--end-day must be >= --start-day")
    if int(args.window_days) <= 0:
        raise SystemExit("--window-days must be > 0")
    if not (0.0 < float(args.cut_ratio) < 1.0):
        raise SystemExit("--cut-ratio must be in (0, 1)")

    os.makedirs(args.result_dir, exist_ok=True)

    registry = StrategyRegistry()
    registry.auto_discover()

    runner = BacktestRunner(metric=args.metric)

    print(
        f"\n[GARCH tune] range={start_day}..{end_day} | "
        f"window_days={args.window_days} cut_ratio={args.cut_ratio} | "
        f"freq={args.frequency} cadence={cadence_s}s | metric={runner.metric}"
    )
    detailed: list[dict[str, Any]] = []
    per_asset: list[dict[str, Any]] = []

    for asset in args.assets:
        res = tournament_tune_asset(
            runner,
            registry,
            asset=asset,
            frequency=args.frequency,
            strategy_names=list(args.strategies),
            start_day=start_day,
            end_day=end_day,
            window_days=int(args.window_days),
            cut_ratio=float(args.cut_ratio),
            num_sims=int(args.num_sims),
            seed=int(args.seed),
        )
        per_asset.append(res)
        detailed.extend(res["survivors"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.result_dir, f"garch_grid_tune_{ts}.json")
    winners = pick_best_individual_per_asset(detailed, metric=args.metric) if detailed else {}

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "range": {"start_day": start_day.isoformat(), "end_day": end_day.isoformat()},
        "window_days": int(args.window_days),
        "cut_ratio": float(args.cut_ratio),
        "request_cadence_s": cadence_s,
        "best_per_asset": winners,
        "per_asset_tournament": per_asset,
        "survivors_full_range": detailed,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n[Report] Wrote {out_json}")

    if not args.no_draft_yaml and winners:
        draft_doc = build_draft_yaml_document(
            winners=winners,
            frequency=args.frequency,
            metric=args.metric,
            start_day=start_day.isoformat(),
            end_day=end_day.isoformat(),
            window_days=int(args.window_days),
            cut_ratio=float(args.cut_ratio),
            seed=int(args.seed),
            json_report=os.path.abspath(out_json),
        )
        draft_path = (
            args.draft_yaml.strip()
            or os.path.join(
                args.result_dir, f"strategies_draft_garch_grid_{ts}.yaml"
            )
        )
        _draft_dir = os.path.dirname(os.path.abspath(draft_path))
        if _draft_dir:
            os.makedirs(_draft_dir, exist_ok=True)
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(
                "# DRAFT — tune_garch_grid tournament: best full-range metric per asset; "
                "same model under all regimes for that asset type (review before merge).\n"
            )
            yaml.safe_dump(
                draft_doc,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print(f"[Draft] Wrote {draft_path}")
        for a, w in sorted(winners.items()):
            print(
                f"  {a}: {w['strategy']} "
                f"(avg_window {args.metric}={w['avg_window_score']:.6f})"
            )
    elif not args.no_draft_yaml and not winners:
        print("[Draft] Skipped — no strategy results to summarize", file=sys.stderr)


if __name__ == "__main__":
    main()
