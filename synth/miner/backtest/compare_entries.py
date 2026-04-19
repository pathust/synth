"""
compare_entries.py — Backtest comparison: entry.py vs entry_old.py (legacy).

Runs both entrypoints on the same random start_time samples and scores
predictions using the same CRPS pipeline as BacktestRunner (cal_reward).

Output layout under --result-dir:
  - ``_compare/`` — full JSON + aggregate plots (``--plot``) after all assets finish
  - ``{ASSET}_{frequency}/`` — saved when that asset finishes; ``compare_progress.json`` for ``--resume``
    and optional incremental checkpoints (``--checkpoint-every N``). With ``--plot``: ``crps_compare_line.png``
    (per-start_time CRPS for entry vs entry_old, not bucket means).
  - ``_compare/compare_run_<run_id>.log`` — optional human-readable progress (``--log-file`` overrides path);
    each line is flushed to disk immediately for ``tail -f`` when SSH disconnects.

Scoring (aligned with prompt config):
  - **low**: ``time_increment=300`` (5m steps), ``time_length=86400``; with ``--start-day``/``--end-day``,
    sample every **12 minutes** (start_time cadence only); bucket CRPS = **mean per UTC calendar day**.
  - **high**: ``time_increment=60`` (1m), ``time_length=3600``; with ``--start-day``/``--end-day``,
    sample every **5 minutes** (start_time cadence only; data remains 1m); bucket CRPS = **mean per UTC hour**.

Real prices for CRPS are cached on disk by ``cal_reward`` (see ``synth/miner/_legacy/backtest_scripts/compute_score.py``).

Example:
    PYTHONPATH=. uv run python -m synth.miner.backtest.compare_entries \
      --assets BTC ETH XAU SOL --frequency low --num-runs 5 --num-sims 100 --window-days 30 \
      --result-dir result/compare_entries --plot

    (--window-days only affects default random date span; simulation history length is
    HISTORY_WINDOW_DAYS in synth/miner/constants.py.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
import importlib
import calendar

# Importing miner modules may pull bittensor, which handles global --help before our argparse.
_SHOW_HELP = False
if __name__ == "__main__" and ("--help" in sys.argv or "-h" in sys.argv):
    _SHOW_HELP = True
    sys.argv = [a for a in sys.argv if a not in ("--help", "-h")]

from synth.db.models import ValidatorRequest
from synth.miner.compute_score import cal_reward
from synth.miner.data_handler import DataHandler
from synth.simulation_input import SimulationInput


class _FlushFileHandler(logging.FileHandler):
    """FileHandler that fsyncs after each record so tail -f sees progress over SSH."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        if self.stream and hasattr(self.stream, "fileno"):
            try:
                os.fsync(self.stream.fileno())
            except (OSError, AttributeError, ValueError):
                pass


def _setup_compare_logging(log_path: str) -> logging.Logger:
    """
    Log to stdout and to log_path (UTC timestamps, flush + fsync each line).
    """
    d = os.path.dirname(os.path.abspath(log_path))
    if d:
        os.makedirs(d, exist_ok=True)

    log = logging.getLogger("synth.miner.backtest.compare_entries")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False

    fmt = logging.Formatter("%(asctime)sZ %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    fmt.converter = time.gmtime

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = _FlushFileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def _utc_floor_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _score_predictions(
    data_handler: DataHandler,
    asset: str,
    start_time: datetime,
    time_increment: int,
    time_length: int,
    predictions,
) -> tuple[float, float | None, float | None]:
    vr = ValidatorRequest(
        asset=asset,
        start_time=start_time,
        time_length=time_length,
        time_increment=time_increment,
    )
    crps, _, _, real_prices = cal_reward(data_handler, vr, predictions)
    if crps == -1:
        return float("inf"), None, None
    # Use real_prices for plotting: first and last non-NaN if possible.
    rp0 = None
    rpn = None
    try:
        if isinstance(real_prices, list) and real_prices:
            # first
            for x in real_prices:
                if x is None:
                    continue
                fx = float(x)
                if fx == fx:
                    rp0 = fx
                    break
            # last
            for x in reversed(real_prices):
                if x is None:
                    continue
                fx = float(x)
                if fx == fx:
                    rpn = fx
                    break
    except Exception:
        rp0 = None
        rpn = None
    return float(crps), rp0, rpn


def _parse_start_time(s: str) -> datetime:
    # Accept "Z" suffix and naive strings (treated as UTC).
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _utc_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")

def _parse_day_utc(s: str) -> date:
    """
    Parse a day string (YYYY-MM-DD) as a UTC date.
    """
    return date.fromisoformat(str(s).strip())

def _parse_hhmm(s: str) -> tuple[int, int]:
    """
    Parse a time-of-day "HH:MM" (24h) into (hour, minute).
    """
    raw = str(s).strip()
    if ":" not in raw:
        raise ValueError(f"Invalid HH:MM time-of-day: {s!r}")
    hh_s, mm_s = raw.split(":", 1)
    hh = int(hh_s)
    mm = int(mm_s)
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid HH:MM time-of-day: {s!r}")
    return hh, mm


def _align_start_time(dt: datetime, *, frequency: str) -> datetime:
    """
    Align sampling granularity:
    - low: align to day boundary (00:00 UTC) so each sample is a "day"
    - high: align to hour boundary (HH:00 UTC) so each sample is an "hour"
    """
    dt = dt.astimezone(timezone.utc)
    if frequency == "low":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(minute=0, second=0, microsecond=0)

def _time_config_for_frequency(frequency: str) -> tuple[int, int]:
    """
    Map prompt frequency label -> (time_increment, time_length).

    - high: 1h horizon, 1m increments
    - low:  1d horizon, 5m increments
    """
    if frequency == "high":
        return 60, 3600
    return 300, 86400


def _sample_start_times(
    start: datetime,
    end: datetime,
    n: int,
    seed: int,
    *,
    frequency: str,
) -> list[datetime]:
    """
    Randomly sample n start times in [start, end], then align by frequency.
    """
    r = random.Random(int(seed))
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    total_seconds = int((end - start).total_seconds())
    if total_seconds <= 0:
        return [_align_start_time(start, frequency=frequency)]

    out: list[datetime] = []
    for _ in range(int(n)):
        dt = start + timedelta(seconds=r.randint(0, total_seconds))
        out.append(_align_start_time(dt, frequency=frequency))
    return out


def _enumerate_start_times(
    start: datetime,
    end: datetime,
    *,
    frequency: str,
) -> list[datetime]:
    """
    Enumerate *all* aligned start times in [start, end] inclusive.
    - low  -> daily grid (00:00 UTC)
    - high -> hourly grid (HH:00 UTC)
    """
    start = _align_start_time(start, frequency=frequency)
    end = _align_start_time(end, frequency=frequency)
    if start > end:
        return []

    step = timedelta(days=1) if frequency == "low" else timedelta(hours=1)
    out: list[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur + step
    return out


def _floor_to_step(dt: datetime, step_seconds: int) -> datetime:
    """
    Floor a UTC datetime down to the nearest step boundary since epoch.
    """
    dt = dt.astimezone(timezone.utc)
    ts = int(dt.timestamp())
    floored = ts - (ts % int(step_seconds))
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _enumerate_cadence_times(
    start: datetime,
    end: datetime,
    *,
    step_seconds: int,
) -> list[datetime]:
    """
    Enumerate start_times on a fixed cadence grid within [start, end) (end exclusive).
    Times are floored to the cadence boundary.
    """
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        return []

    step = timedelta(seconds=int(step_seconds))
    cur = _floor_to_step(start, int(step_seconds))
    if cur < start:
        cur = cur + step

    out: list[datetime] = []
    while cur < end:
        out.append(cur)
        cur = cur + step
    return out

def _enumerate_all_times_in_days(
    start_day: date,
    end_day: date,
    *,
    frequency: str,
    day_start_hhmm: str | None = None,
    day_end_hhmm: str | None = None,
) -> list[datetime]:
    """
    Enumerate all start_times within [start_day, end_day] inclusive.

    Enumerate start_times at a fixed cadence per frequency:
    - high: every 5 minutes (300s)  — start_time cadence only; data remains 1m
    - low:  every 12 minutes (720s) — start_time cadence only; data remains 5m
    """
    if end_day < start_day:
        return []

    start_dt = datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc)
    end_exclusive = datetime(end_day.year, end_day.month, end_day.day, tzinfo=timezone.utc) + timedelta(days=1)

    # Optional intra-day window (end is exclusive)
    if day_start_hhmm:
        hh, mm = _parse_hhmm(day_start_hhmm)
        start_dt = start_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if day_end_hhmm:
        hh, mm = _parse_hhmm(day_end_hhmm)
        # end time-of-day refers to the end_day's clock; exclusive bound.
        end_exclusive = datetime(end_day.year, end_day.month, end_day.day, tzinfo=timezone.utc).replace(
            hour=hh, minute=mm, second=0, microsecond=0
        )
        # If end <= start and caller passed same day, yield empty.
        if end_exclusive <= start_dt:
            return []

    step_seconds = 300 if frequency == "high" else 720
    step = timedelta(seconds=step_seconds)
    out: list[datetime] = []
    cur = start_dt
    while cur < end_exclusive:
        out.append(cur)
        cur = cur + step
    return out


def _add_months(d: date, months: int) -> date:
    """Add months to a date, clamping day to month length."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = calendar.monthrange(y, m)[1]
    day = min(d.day, last)
    return date(y, m, day)


def _month_bounds(ym: date) -> tuple[date, date]:
    """Return (first_day, last_day) for ym's month."""
    first = date(ym.year, ym.month, 1)
    last_day = calendar.monthrange(ym.year, ym.month)[1]
    last = date(ym.year, ym.month, last_day)
    return first, last


def _sample_days_last_n_months(
    *,
    end_day: date,
    n_months: int,
    days_per_month: int,
    seed: int,
) -> list[date]:
    """
    Sample ``days_per_month`` distinct calendar days from each of the last ``n_months``
    ending at ``end_day`` (inclusive). For the current month, the eligible range is
    [month_start, end_day]. For prior months, eligible range is full month.
    """
    if n_months <= 0 or days_per_month <= 0:
        return []

    out: list[date] = []
    # Anchor on the month that contains end_day, then step backwards.
    anchor = date(end_day.year, end_day.month, 1)
    for i in range(int(n_months)):
        month_anchor = _add_months(anchor, -i)
        m_start, m_end = _month_bounds(month_anchor)
        if i == 0:
            m_end = min(m_end, end_day)
        # Build candidate days in this month window.
        n_days = (m_end - m_start).days + 1
        if n_days <= 0:
            continue
        candidates = [m_start + timedelta(days=j) for j in range(n_days)]
        r = random.Random(int(seed) + 1009 * i)
        k = min(int(days_per_month), len(candidates))
        picks = r.sample(candidates, k=k)
        out.extend(sorted(picks))

    # Sort chronologically for stable output.
    out = sorted(out)
    return out


def _scheduled_dates(
    args: argparse.Namespace,
    *,
    start: datetime,
    end: datetime,
    frequency: str,
) -> list[datetime]:
    if args.start_time:
        return [_align_start_time(_parse_start_time(str(args.start_time)), frequency=frequency)]
    if getattr(args, "sample_last_n_months", 0):
        end_day = end.astimezone(timezone.utc).date()
        days = _sample_days_last_n_months(
            end_day=end_day,
            n_months=int(args.sample_last_n_months),
            days_per_month=int(args.days_per_month),
            seed=int(args.seed),
        )
        out: list[datetime] = []
        for d in days:
            out.extend(
                _enumerate_all_times_in_days(
                    d,
                    d,
                    frequency=frequency,
                    day_start_hhmm=(str(args.day_start_hhmm) if args.day_start_hhmm else None),
                    day_end_hhmm=(str(args.day_end_hhmm) if args.day_end_hhmm else None),
                )
            )
        return out
    if args.start_day or args.end_day:
        if not (args.start_day and args.end_day):
            raise SystemExit("Must provide both --start-day and --end-day")
        sd = _parse_day_utc(str(args.start_day))
        ed = _parse_day_utc(str(args.end_day))
        return _enumerate_all_times_in_days(
            sd,
            ed,
            frequency=frequency,
            day_start_hhmm=(str(args.day_start_hhmm) if args.day_start_hhmm else None),
            day_end_hhmm=(str(args.day_end_hhmm) if args.day_end_hhmm else None),
        )
    # If user explicitly provided a datetime window, enumerate on fixed cadence
    # (high=5m, low=12m) within [start, end) regardless of --all/--num-runs.
    if args.start_date and args.end_date:
        step_seconds = 300 if frequency == "high" else 720
        return _enumerate_cadence_times(start, end, step_seconds=step_seconds)
    if args.all:
        return _enumerate_start_times(start, end, frequency=frequency)
    return _sample_start_times(
        start,
        end,
        int(args.num_runs),
        int(args.seed),
        frequency=frequency,
    )


def _bucket_start(dt: datetime, *, frequency: str) -> datetime:
    """
    Bucket key for averaging CRPS:
    - high: per-hour mean CRPS (HH:00 UTC)
    - low:  per-day mean CRPS (00:00 UTC)
    """
    dt = dt.astimezone(timezone.utc)
    if frequency == "high":
        return dt.replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _asset_subdir(asset: str, frequency: str) -> str:
    """One folder per (asset, frequency), e.g. BTC_high, AAPLX_low."""
    return f"{asset}_{frequency}"


def _compare_subdir() -> str:
    """Aggregate comparison folder under --result-dir."""
    return "_compare"


def _as_float_or_nan(x: object) -> float:
    if x is None:
        return float("nan")
    return float(x)


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x == x and x != float("inf")]
    return float(statistics.fmean(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x == x and x != float("inf")]
    return float(statistics.median(xs)) if xs else None


def _fingerprint(
    args: argparse.Namespace,
    *,
    asset: str,
    frequency: str,
    dates: list[datetime],
) -> dict[str, Any]:
    ti, tl = _time_config_for_frequency(frequency)
    return {
        "asset": asset,
        "frequency": frequency,
        "time_increment": ti,
        "time_length": tl,
        "num_sims": int(args.num_sims),
        "seed": int(args.seed),
        "sample_last_n_months": int(getattr(args, "sample_last_n_months", 0) or 0),
        "days_per_month": int(getattr(args, "days_per_month", 0) or 0),
        "start_day": args.start_day,
        "end_day": args.end_day,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "all": bool(args.all),
        "start_time": args.start_time,
        "window_days": int(args.window_days),
        "n_scheduled_dates": len(dates),
        "first_date": dates[0].isoformat() if dates else None,
        "last_date": dates[-1].isoformat() if dates else None,
    }


def _progress_path(asset_dir: str) -> str:
    return os.path.join(asset_dir, "compare_progress.json")


def _load_cached_rows(asset_dir: str, fp: dict[str, Any], *, resume: bool) -> list[dict]:
    if not resume:
        return []
    path = _progress_path(asset_dir)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if data.get("fingerprint") != fp:
        return []
    return list(data.get("rows") or [])


def _save_progress(asset_dir: str, fp: dict[str, Any], rows: list[dict], run_id: str) -> None:
    os.makedirs(asset_dir, exist_ok=True)
    path = _progress_path(asset_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"fingerprint": fp, "run_id": run_id, "rows": rows},
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp, path)


def _bucket_rows_from_rows(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(str(r["bucket_start"]), []).append(r)
    out: list[dict] = []
    for bstart in sorted(buckets.keys()):
        rs = buckets[bstart]
        entry_b = _mean([float(rr["entry_crps"]) for rr in rs])
        legacy_b = _mean([float(rr["entry_legacy_crps"]) for rr in rs])
        if entry_b is None or legacy_b is None:
            continue
        out.append(
            {
                "asset": str(rs[0]["asset"]),
                "frequency": str(rs[0]["frequency"]),
                "bucket_start": bstart,
                "n_samples": len(rs),
                "entry_crps_mean": float(entry_b),
                "legacy_crps_mean": float(legacy_b),
                "delta_legacy_minus_entry": float(legacy_b) - float(entry_b),
            }
        )
    return out


def _per_asset_stats(
    rows: list[dict],
    bucket_rows: list[dict],
    *,
    asset: str,
    frequency: str,
    eps: float,
) -> dict[str, Any]:
    d = [float(rr["delta_crps_legacy_minus_entry"]) for rr in rows]
    win_legacy_a = sum(1 for x in d if x < -eps)
    win_entry_a = sum(1 for x in d if x > eps)
    tie_a = sum(1 for x in d if abs(x) <= eps)
    denom_a = len(d)
    b_rows = [br for br in bucket_rows if br["asset"] == asset and br["frequency"] == frequency]
    b_deltas = [float(br["delta_legacy_minus_entry"]) for br in b_rows]
    return {
        "asset": asset,
        "frequency": frequency,
        "candle_step_seconds": 60 if frequency == "high" else 300,
        "crps_bucket": "utc_hour_mean" if frequency == "high" else "utc_day_mean",
        "n_samples": denom_a,
        "wins_legacy": win_legacy_a,
        "wins_entry": win_entry_a,
        "ties": tie_a,
        "win_rate_legacy": win_legacy_a / denom_a if denom_a else None,
        "win_rate_entry": win_entry_a / denom_a if denom_a else None,
        "tie_rate": tie_a / denom_a if denom_a else None,
        "mean_delta_legacy_minus_entry": _mean(d),
        "median_delta_legacy_minus_entry": _median(d),
        "mean_entry_crps": _mean([float(rr["entry_crps"]) for rr in rows]),
        "mean_legacy_crps": _mean([float(rr["entry_legacy_crps"]) for rr in rows]),
        "n_buckets": len(b_rows),
        "bucket_win_rate_legacy": (sum(1 for x in b_deltas if x < -eps) / len(b_deltas)) if b_deltas else None,
        "bucket_win_rate_entry": (sum(1 for x in b_deltas if x > eps) / len(b_deltas)) if b_deltas else None,
        "bucket_tie_rate": (sum(1 for x in b_deltas if abs(x) <= eps) / len(b_deltas)) if b_deltas else None,
        "bucket_mean_entry_crps": _mean([float(br["entry_crps_mean"]) for br in b_rows]),
        "bucket_mean_legacy_crps": _mean([float(br["legacy_crps_mean"]) for br in b_rows]),
        "bucket_mean_delta_legacy_minus_entry": _mean(b_deltas) if b_deltas else None,
    }


def _rows_sorted_by_start_time(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: str(r.get("start_time", "")))


def _crps_scalar_for_plot(x: object) -> float:
    """Finite floats as-is; inf / bad -> NaN so the line breaks instead of spiking."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if v == float("inf") or v == float("-inf") or v != v:
        return float("nan")
    return v


def _plot_one_asset(
    asset_dir: str,
    asset: str,
    frequency: str,
    rows: list[dict],
    *,
    entry_label: str = "entry",
    legacy_label: str = "entry_old",
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_s = _rows_sorted_by_start_time(rows)
    n = len(rows_s)
    xs = list(range(n))
    entry_vals = [_crps_scalar_for_plot(r.get("entry_crps")) for r in rows_s]
    legacy_vals = [_crps_scalar_for_plot(r.get("entry_legacy_crps")) for r in rows_s]
    price_vals = []
    for r in rows_s:
        p = r.get("real_price_start")
        try:
            fp = float(p)
        except (TypeError, ValueError):
            fp = float("nan")
        if fp == float("inf") or fp == float("-inf"):
            fp = float("nan")
        price_vals.append(fp)

    width = max(10, min(28, 0.006 * n + 8))
    title = f"CRPS per sample ({entry_label} vs {legacy_label}), n={n}"
    xlabel = "sample index (sorted by start_time)"
    lw = 0.9 if n > 300 else 1.2

    fig = plt.figure(figsize=(width, 5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    if not rows_s:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    elif n > 300:
        ax.plot(xs, entry_vals, label=entry_label, lw=lw)
        ax.plot(xs, legacy_vals, label=legacy_label, lw=lw)
        ax.set_title(title)
    else:
        ax.plot(xs, entry_vals, label=entry_label, marker="o", ms=2, lw=lw)
        ax.plot(xs, legacy_vals, label=legacy_label, marker="o", ms=2, lw=lw)
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CRPS (lower is better)")
    ax.grid(alpha=0.25)
    # Secondary axis: real price at start_time (spot), aligned to sample index.
    if any(v == v for v in price_vals):
        ax2 = ax.twinx()
        ax2.plot(xs, price_vals, color="#777777", alpha=0.45, lw=0.9, label="real_price_start")
        ax2.set_ylabel("Real price (spot)")
        # Merge legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax.legend()
    fig.tight_layout()
    p_line = os.path.join(asset_dir, "crps_compare_line.png")
    fig.savefig(p_line)
    plt.close(fig)

    plots_a: dict[str, str] = {"crps_compare_line": p_line}

    return plots_a


def _missing_plot_files(asset_dir: str) -> list[str]:
    expected = ["crps_compare_line.png"]
    return [name for name in expected if not os.path.isfile(os.path.join(asset_dir, name))]


def _load_entry_callable(spec: str):
    """
    Load an entry function from a spec like:
      - synth.miner.entry:generate_simulations
      - synth.miner.entry_old:generate_simulations_legacy
    """
    s = str(spec).strip()
    if ":" not in s:
        raise ValueError(
            f"Invalid entry spec {spec!r}. Expected 'module:function', e.g. "
            "'synth.miner.entry:generate_simulations'."
        )
    mod_name, fn_name = s.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"Entry {spec!r} is not callable")
    label = mod_name.rsplit(".", 1)[-1]
    if fn_name != "generate_simulations":
        label = f"{label}.{fn_name}"
    return fn, label


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two entry modules (backtest)")
    p.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Assets to run (ignored when --all-assets-default is set).",
    )
    p.add_argument(
        "--frequency",
        choices=["high", "low"],
        default="low",
        help="Frequency for all assets (ignored when --all-assets-default is set).",
    )
    p.add_argument(
        "--all-assets-default",
        action="store_true",
        help=(
            "Run the default asset lists from run_strategy_scan.py: "
            "ALL_ASSETS_LOW (low) + ALL_ASSETS_HIGH (high). "
            "This runs both frequencies in one invocation."
        ),
    )
    p.add_argument(
        "--all-assets-low-default",
        action="store_true",
        help="Run only ALL_ASSETS_LOW (low) from run_strategy_scan.py.",
    )
    p.add_argument(
        "--all-assets-high-default",
        action="store_true",
        help="Run only ALL_ASSETS_HIGH (high) from run_strategy_scan.py.",
    )
    p.add_argument(
        "--skip-pairs",
        nargs="+",
        default=None,
        metavar="ASSET_FREQUENCY",
        help=(
            "Skip specific asset-frequency pairs, e.g. BTC_low ETH_low. "
            "Useful when some pairs already finished."
        ),
    )
    p.add_argument(
        "--start-date",
        default=None,
        help=(
            "Override sampling window start (ISO8601, e.g. 2026-03-01T00:00:00Z). "
            "If omitted, uses (now_utc_hour - 2d - window_days)."
        ),
    )
    p.add_argument(
        "--sample-last-n-months",
        type=int,
        default=0,
        metavar="N",
        help=(
            "New mode: sample random days_per_month days from each of the last N months "
            "(ending at default end_date = now_utc_hour - 2d unless --end-date is set). "
            "Month N=4 with days_per_month=5 yields ~20 days. "
            "This mode enumerates start_times within each picked day using the same cadence "
            "as --start-day/--end-day (high=5m, low=12m)."
        ),
    )
    p.add_argument(
        "--days-per-month",
        type=int,
        default=5,
        metavar="K",
        help="With --sample-last-n-months: how many distinct days to sample per month (default: 5).",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help=(
            "Override sampling window end (ISO8601, e.g. 2026-04-01T00:00:00Z). "
            "If omitted, uses (now_utc_hour - 2d)."
        ),
    )
    p.add_argument(
        "--start-day",
        default=None,
        help=(
            "If set with --end-day: enumerate start_times within these UTC days (YYYY-MM-DD) "
            "at a fixed cadence (high=5m, low=12m)."
        ),
    )
    p.add_argument(
        "--end-day",
        default=None,
        help=(
            "If set with --start-day: enumerate start_times within these UTC days (YYYY-MM-DD) "
            "at a fixed cadence (high=5m, low=12m)."
        ),
    )
    p.add_argument(
        "--day-start-hhmm",
        default=None,
        metavar="HH:MM",
        help=(
            "Optional (only with --start-day/--end-day): limit start_times to >= this UTC time-of-day, "
            "e.g. 10:00."
        ),
    )
    p.add_argument(
        "--day-end-hhmm",
        default=None,
        metavar="HH:MM",
        help=(
            "Optional (only with --start-day/--end-day): limit start_times to < this UTC time-of-day "
            "(exclusive), e.g. 11:00."
        ),
    )
    p.add_argument(
        "--start-time",
        default=None,
        help=(
            "If set: run a single fixed start_time instead of random sampling "
            "(ISO8601, e.g. 2026-04-01T00:00:00Z)."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "If set (and --start-time not set): enumerate ALL start_times in the window "
            "(low=daily, high=hourly) instead of random sampling."
        ),
    )
    p.add_argument("--num-runs", type=int, default=5, help="Random start_time samples per asset")
    p.add_argument("--num-sims", type=int, default=100, help="num_simulations passed to entry")
    p.add_argument(
        "--window-days",
        type=int,
        default=30,
        help=(
            "Only for random / --all sampling: span (days) before default end when "
            "--start-date/--end-date omitted. Does not limit simulation history; that is "
            "synth.miner.constants.HISTORY_WINDOW_DAYS."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--result-dir",
        default="result/compare_entries",
        help=(
            "Root output directory. Writes per-asset folders (ASSET_freq/) and "
            "aggregate under _compare/."
        ),
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="If set, write CRPS plots into each asset folder and into _compare/.",
    )
    p.add_argument(
        "--entry-a",
        default="synth.miner.entry:generate_simulations",
        metavar="MODULE:FUNC",
        help=(
            "Entry A callable spec. Format 'module:function'. "
            "Default: synth.miner.entry:generate_simulations"
        ),
    )
    p.add_argument(
        "--entry-b",
        default="synth.miner.entry_old:generate_simulations_legacy",
        metavar="MODULE:FUNC",
        help=(
            "Entry B callable spec. Format 'module:function'. "
            "Default: synth.miner.entry_old:generate_simulations_legacy"
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse rows from ASSET_freq/compare_progress.json when fingerprint matches "
            "(same dates, num_sims, seed, window, etc.); only missing start_times are recomputed."
        ),
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If >0: rewrite compare_progress.json every N newly computed rows (crash-safe). "
            "0 = only checkpoint at end of each asset."
        ),
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Append progress lines here with immediate flush (safe for tail -f after SSH drops). "
            "Default: <result-dir>/_compare/compare_run_<run_id>.log"
        ),
    )
    args = p.parse_args()
    if _SHOW_HELP:
        p.print_help()
        raise SystemExit(0)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    compare_root = os.path.join(args.result_dir, _compare_subdir())

    default_end = _utc_floor_hour(datetime.now(timezone.utc)) - timedelta(days=2)
    default_start = default_end - timedelta(days=int(args.window_days))
    end = _parse_start_time(str(args.end_date)) if args.end_date else default_end
    start = _parse_start_time(str(args.start_date)) if args.start_date else default_start
    if start > end:
        raise SystemExit("--start-date must be <= --end-date")

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(compare_root, exist_ok=True)

    log_path = args.log_file or os.path.join(compare_root, f"compare_run_{run_id}.log")
    logger = _setup_compare_logging(log_path)
    logger.info("compare_entries run_id=%s log_file=%s", run_id, os.path.abspath(log_path))
    logger.info("result_dir=%s", os.path.abspath(args.result_dir))

    data_handler = DataHandler()

    entry_generate, entry_label = _load_entry_callable(args.entry_a)
    entry_legacy_generate, legacy_label = _load_entry_callable(args.entry_b)
    logger.info("entry_a=%s label=%s", args.entry_a, entry_label)
    logger.info("entry_b=%s label=%s", args.entry_b, legacy_label)

    results: list[dict] = []
    bucket_rows: list[dict] = []
    per_asset: dict[str, dict[str, Any]] = {}
    per_asset_paths: dict[str, str] = {}
    per_asset_plot_paths: dict[str, Any] = {}
    eps = 1e-12

    selected_default_modes = sum(
        1
        for x in [
            args.all_assets_default,
            args.all_assets_low_default,
            args.all_assets_high_default,
        ]
        if x
    )
    if selected_default_modes > 1:
        raise SystemExit(
            "Use only one of --all-assets-default, --all-assets-low-default, --all-assets-high-default"
        )

    if args.all_assets_default or args.all_assets_low_default or args.all_assets_high_default:
        from synth.miner.run_strategy_scan import ALL_ASSETS_HIGH, ALL_ASSETS_LOW

        pairs: list[tuple[str, str]] = []
        if args.all_assets_default or args.all_assets_low_default:
            for a in ALL_ASSETS_LOW:
                pairs.append((a, "low"))
        if args.all_assets_default or args.all_assets_high_default:
            for a in ALL_ASSETS_HIGH:
                pairs.append((a, "high"))
    else:
        if not args.assets:
            raise SystemExit(
                "Must provide --assets or set one of "
                "--all-assets-default/--all-assets-low-default/--all-assets-high-default"
            )
        pairs = [(a, str(args.frequency)) for a in args.assets]

    skip_pairs = {str(x).strip() for x in (args.skip_pairs or []) if str(x).strip()}
    skipped_pairs_for_compute: list[tuple[str, str]] = []
    if skip_pairs:
        pairs_before = pairs
        pairs = [(asset, freq) for asset, freq in pairs if f"{asset}_{freq}" not in skip_pairs]
        skipped_pairs_for_compute = [
            (asset, freq) for asset, freq in pairs_before if f"{asset}_{freq}" in skip_pairs
        ]
        skipped = [f"{asset}_{freq}" for asset, freq in skipped_pairs_for_compute]
        if skipped:
            logger.info("[skip-pairs] skipped=%d -> %s", len(skipped), ", ".join(sorted(skipped)))
    if not pairs and not (args.plot and skipped_pairs_for_compute):
        raise SystemExit("No asset-frequency pairs to run after filtering")

    # Plot skipped pairs from cache immediately (do not wait for the long main loop).
    if args.plot and skipped_pairs_for_compute:
        for asset, frequency in skipped_pairs_for_compute:
            key = f"{asset}::{frequency}"
            dates = _scheduled_dates(args, start=start, end=end, frequency=frequency)
            fp = _fingerprint(args, asset=asset, frequency=frequency, dates=dates)
            asset_dir = os.path.join(args.result_dir, _asset_subdir(asset, frequency))
            cached = _load_cached_rows(asset_dir, fp, resume=True)
            if not cached:
                logger.info("[plot-from-cache] %s: no matching cache rows", key)
                continue

            buckets_a = _bucket_rows_from_rows(cached)
            st_one = _per_asset_stats(cached, buckets_a, asset=asset, frequency=frequency, eps=eps)
            per_asset[key] = st_one
            missing_plots = _missing_plot_files(asset_dir)
            if not missing_plots:
                logger.info("[plot-skip] %s: plots already exist", key)
                continue
            try:
                plots_a = _plot_one_asset(
                    asset_dir,
                    asset,
                    frequency,
                    cached,
                    entry_label=entry_label,
                    legacy_label=legacy_label,
                )
                per_asset_plot_paths[key] = plots_a
                logger.info(
                    "[plot-from-cache] %s: wrote %s",
                    key,
                    ", ".join(missing_plots),
                )
            except Exception as e:
                per_asset_plot_paths[key] = {"error": str(e)[:200]}

    for asset, frequency in pairs:
        time_increment, time_length = _time_config_for_frequency(frequency)
        dates = _scheduled_dates(args, start=start, end=end, frequency=frequency)

        key = f"{asset}::{frequency}"
        asset_dir = os.path.join(args.result_dir, _asset_subdir(asset, frequency))
        os.makedirs(asset_dir, exist_ok=True)
        fp = _fingerprint(args, asset=asset, frequency=frequency, dates=dates)
        cached = _load_cached_rows(asset_dir, fp, resume=bool(args.resume))
        by_st = {str(r["start_time"]): r for r in cached}
        merged: list[dict] = []
        new_computed = 0
        n_hit = sum(1 for d in dates if d.isoformat() in by_st)
        if args.resume and n_hit:
            logger.info("[resume] %s: %d/%d start_times from compare_progress.json", key, n_hit, len(dates))

        logger.info(
            "[%s] total=%d already_cached=%d will_compute=%d",
            key,
            len(dates),
            n_hit,
            len(dates) - n_hit,
        )

        for i, dt in enumerate(dates):
            dt_utc = dt.astimezone(timezone.utc)
            st_key = _utc_iso_z(dt_utc)
            bucket_dt = _bucket_start(dt_utc, frequency=frequency)
            bucket_key = _utc_iso_z(bucket_dt)
            if st_key in by_st:
                merged.append(by_st[st_key])
                continue

            si = SimulationInput(
                asset=asset,
                start_time=st_key,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=int(args.num_sims),
            )

            row: dict = {
                "asset": asset,
                "frequency": frequency,
                "start_time": st_key,
                "start_time_unix": int(dt_utc.timestamp()),
                "bucket_start": bucket_key,
                "bucket_start_unix": int(bucket_dt.timestamp()),
                "time_increment": time_increment,
                "time_length": time_length,
                "num_simulations": int(args.num_sims),
                "seed": int(args.seed) + i,
            }

            try:
                out_a = entry_generate(
                    simulation_input=si,
                    asset=asset,
                    start_time=si.start_time,
                    time_increment=time_increment,
                    time_length=time_length,
                    num_simulations=int(args.num_sims),
                    seed=int(args.seed) + i,
                )
                pred_a = out_a.get("predictions")
                row["entry_ok"] = pred_a is not None
                if pred_a is not None:
                    crps_a, rp0_a, rpn_a = _score_predictions(
                        data_handler, asset, dt, time_increment, time_length, pred_a
                    )
                    row["entry_crps"] = crps_a
                    row["real_price_start"] = rp0_a
                    row["real_price_end"] = rpn_a
                else:
                    row["entry_crps"] = float("inf")
                    row["real_price_start"] = None
                    row["real_price_end"] = None
            except Exception as e:
                row["entry_ok"] = False
                row["entry_error"] = str(e)[:200]
                row["entry_crps"] = float("inf")
                row["real_price_start"] = None
                row["real_price_end"] = None

            try:
                out_b = entry_legacy_generate(
                    simulation_input=si,
                    asset=asset,
                    start_time=si.start_time,
                    time_increment=time_increment,
                    time_length=time_length,
                    num_simulations=int(args.num_sims),
                    seed=int(args.seed) + i,
                )
                pred_b = out_b.get("predictions")
                row["entry_legacy_ok"] = pred_b is not None
                row["entry_legacy_crps"] = (
                    _score_predictions(data_handler, asset, dt, time_increment, time_length, pred_b)[0]
                    if pred_b is not None
                    else float("inf")
                )
            except Exception as e:
                row["entry_legacy_ok"] = False
                row["entry_legacy_error"] = str(e)[:200]
                row["entry_legacy_crps"] = float("inf")

            row["delta_crps_legacy_minus_entry"] = float(row["entry_legacy_crps"]) - float(row["entry_crps"])
            merged.append(row)
            by_st[st_key] = row
            new_computed += 1

            logger.info(
                "%s %s start=%s bucket=%s  %s=%.4f  %s=%.4f  Δ=%.4f",
                asset,
                frequency,
                st_key,
                bucket_key,
                entry_label,
                float(row["entry_crps"]),
                legacy_label,
                float(row["entry_legacy_crps"]),
                float(row["delta_crps_legacy_minus_entry"]),
            )

            ce = int(args.checkpoint_every)
            if ce > 0 and new_computed % ce == 0:
                _save_progress(asset_dir, fp, merged, run_id)

        _save_progress(asset_dir, fp, merged, run_id)

        results.extend(merged)
        buckets_a = _bucket_rows_from_rows(merged)
        bucket_rows.extend(buckets_a)
        st_one = _per_asset_stats(merged, buckets_a, asset=asset, frequency=frequency, eps=eps)
        per_asset[key] = st_one

        payload = {
            "run_id": run_id,
            "fingerprint": fp,
            "asset": asset,
            "frequency": frequency,
            "summary": st_one,
            "bucket_rows": buckets_a,
            "rows": merged,
        }
        apath = os.path.join(asset_dir, f"compare_entries_{run_id}.json")
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        per_asset_paths[key] = apath
        logger.info("[saved] %s -> %s (%d rows)", key, apath, len(merged))

        if args.plot:
            try:
                missing_plots = _missing_plot_files(asset_dir)
                if new_computed > 0 or missing_plots:
                    plots_a = _plot_one_asset(
                        asset_dir,
                        asset,
                        frequency,
                        merged,
                        entry_label=entry_label,
                        legacy_label=legacy_label,
                    )
                    per_asset_plot_paths[key] = plots_a
                    if missing_plots and new_computed == 0:
                        logger.info(
                            "[plot-from-cache] %s: generated missing plots -> %s",
                            key,
                            ", ".join(missing_plots),
                        )
                else:
                    logger.info("[plot-skip] %s: plots already exist", key)
            except Exception as e:
                per_asset_plot_paths[key] = {"error": str(e)[:200]}

    # Global summary (all assets)
    deltas = [float(r["delta_crps_legacy_minus_entry"]) for r in results]
    entry_scores = [float(r["entry_crps"]) for r in results]
    legacy_scores = [float(r["entry_legacy_crps"]) for r in results]

    win_legacy = sum(1 for d in deltas if d < -eps)
    win_entry = sum(1 for d in deltas if d > eps)
    tie = sum(1 for d in deltas if abs(d) <= eps)
    denom = len(deltas)

    bucket_deltas = [float(r["delta_legacy_minus_entry"]) for r in bucket_rows]
    bucket_win_legacy = sum(1 for d in bucket_deltas if d < -eps)
    bucket_win_entry = sum(1 for d in bucket_deltas if d > eps)
    bucket_tie = sum(1 for d in bucket_deltas if abs(d) <= eps)
    bucket_denom = len(bucket_deltas)

    unique_keys = sorted(per_asset.keys())

    summary = {
        "n": int(len(results)),
        "mean_entry_crps": _mean(entry_scores),
        "mean_legacy_crps": _mean(legacy_scores),
        "mean_delta_legacy_minus_entry": _mean(deltas),
        "median_delta_legacy_minus_entry": _median(deltas),
        "wins_legacy": win_legacy,
        "wins_entry": win_entry,
        "ties": tie,
        "win_rate_legacy": (win_legacy / denom) if denom else None,
        "win_rate_entry": (win_entry / denom) if denom else None,
        "tie_rate": (tie / denom) if denom else None,
        "bucketed": {
            "n": bucket_denom,
            "mean_entry_crps": _mean([float(r["entry_crps_mean"]) for r in bucket_rows]) if bucket_rows else None,
            "mean_legacy_crps": _mean([float(r["legacy_crps_mean"]) for r in bucket_rows]) if bucket_rows else None,
            "mean_delta_legacy_minus_entry": _mean(bucket_deltas) if bucket_deltas else None,
            "median_delta_legacy_minus_entry": _median(bucket_deltas) if bucket_deltas else None,
            "wins_legacy": bucket_win_legacy,
            "wins_entry": bucket_win_entry,
            "ties": bucket_tie,
            "win_rate_legacy": (bucket_win_legacy / bucket_denom) if bucket_denom else None,
            "win_rate_entry": (bucket_win_entry / bucket_denom) if bucket_denom else None,
            "tie_rate": (bucket_tie / bucket_denom) if bucket_denom else None,
            "bucket_definition": {
                "high": "mean CRPS per UTC hour (bucket_start = HH:00)",
                "low": "mean CRPS per UTC calendar day (bucket_start = 00:00)",
            },
            "scoring_reference": {
                "low": {
                    "time_increment_s": 300,
                    "time_length_s": 86400,
                    "start_day_range_step_s": 720,
                },
                "high": {
                    "time_increment_s": 60,
                    "time_length_s": 3600,
                    "start_day_range_step_s": 300,
                },
            },
        },
        "window_days": int(args.window_days),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "start_day": (str(args.start_day) if args.start_day else None),
        "end_day": (str(args.end_day) if args.end_day else None),
        "num_runs_per_asset": (
            1
            if args.start_time
            else ("all" if args.all else int(args.num_runs))
        ),
        "num_sims": int(args.num_sims),
        "mode": (
            "all_assets_default"
            if args.all_assets_default
            else (
                "all_assets_low_default"
                if args.all_assets_low_default
                else ("all_assets_high_default" if args.all_assets_high_default else "manual")
            )
        ),
        "pairs": pairs,
        "fixed_start_time": (str(args.start_time) if args.start_time else None),
        "all_mode": bool(args.all),
        "per_asset": per_asset,
        "run_id": run_id,
        "output": {
            "root": os.path.abspath(args.result_dir),
            "compare_all": os.path.abspath(compare_root),
            "per_asset_pattern": os.path.join(os.path.abspath(args.result_dir), "{ASSET}_{frequency}"),
        },
    }

    summary["resume_enabled"] = bool(args.resume)
    summary["checkpoint_every"] = int(args.checkpoint_every)
    summary["per_asset_output_json"] = per_asset_paths
    if per_asset_plot_paths:
        summary["per_asset_plots"] = per_asset_plot_paths

    # Optional plotting: aggregate under _compare/
    plot_paths: dict[str, str] = {}
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            keys = [k for k in unique_keys if k in per_asset]
            if keys:
                keys_sorted = sorted(
                    keys,
                    key=lambda k: (per_asset[k].get("frequency", ""), per_asset[k].get("asset", "")),
                )
                x = list(range(len(keys_sorted)))
                entry_mean = [_as_float_or_nan(per_asset[k].get("mean_entry_crps")) for k in keys_sorted]
                legacy_mean = [_as_float_or_nan(per_asset[k].get("mean_legacy_crps")) for k in keys_sorted]

                fig = plt.figure(figsize=(max(10, len(keys_sorted) * 0.8), 6), dpi=140)
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(x, entry_mean, label=entry_label, marker="o", ms=4)
                ax.plot(x, legacy_mean, label=legacy_label, marker="o", ms=4)
                ax.set_title(f"Mean CRPS by asset ({entry_label} vs {legacy_label})")
                ax.set_ylabel("CRPS (lower is better)")
                ax.set_xticks(x)
                ax.set_xticklabels(keys_sorted, rotation=45, ha="right")
                ax.grid(alpha=0.25)
                ax.legend()
                fig.tight_layout()
                out_png = os.path.join(compare_root, "crps_compare_by_asset.png")
                fig.savefig(out_png)
                plt.close(fig)
                plot_paths["crps_compare_by_asset"] = out_png
        except Exception as e:
            summary["plot_error"] = str(e)[:200]

    if plot_paths:
        summary["plots"] = plot_paths

    out_path = os.path.join(compare_root, f"compare_entries_{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False, indent=2)

    logger.info("\n=== SUMMARY ===\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Wrote (all assets): %s", out_path)
    if per_asset_paths:
        logger.info("Per-asset JSON:")
        for k, pth in sorted(per_asset_paths.items()):
            logger.info("  %s -> %s", k, pth)


if __name__ == "__main__":
    main()

