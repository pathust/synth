"""
Build a draft strategies YAML after tuning: copy the live file and patch only
rows that have per-regime tuning results.

Regime keys may come from legacy pattern (bullish/bearish/neutral) or from
``detect_regime`` production labels (bull/high_vol/ranging, trending/mean_reverting, …).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Backtest scanner / pattern bias → keys in strategies.yaml (crypto)
SCANNER_TO_YAML_REGIME: dict[str, str] = {
    "bullish": "bull",
    "bearish": "high_vol",
    "neutral": "ranging",
}

# Keys from synth.miner.regimes.detector (already match strategies.yaml)
_PRODUCTION_YAML_KEYS: frozenset[str] = frozenset(
    {
        "bull",
        "high_vol",
        "ranging",
        "trending",
        "mean_reverting",
        "market_open",
        "overnight",
        "earnings",
    }
)


def regime_label_to_yaml_key(label: str) -> str | None:
    s = str(label).lower()
    if s in SCANNER_TO_YAML_REGIME:
        return SCANNER_TO_YAML_REGIME[s]
    if s in _PRODUCTION_YAML_KEYS:
        return s
    return None


def _default_strategies_yaml_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "strategies.yaml"


def apply_tune_regimes_to_routing(
    doc: dict[str, Any],
    tuning_results: list[dict[str, Any]],
    *,
    rank_slot: int = 1,
) -> int:
    """
    Mutate doc['routing'] in place. Only entries with use_regimes + regime_results are applied.

    When multiple tuning runs exist for the same asset×frequency (e.g. one row per strategy),
    only results with matching rank_slot are applied (default: 1).

    Returns number of regime cells updated.
    """
    n = 0
    routing = doc.setdefault("routing", {})
    for tr in tuning_results:
        if tr.get("rank_slot", 1) != rank_slot:
            continue
        if not tr.get("use_regimes") or not tr.get("regime_results"):
            continue
        asset = tr.get("asset")
        freq = tr.get("frequency")
        strat = tr.get("strategy")
        if not asset or not freq or not strat:
            continue
        regime_results = tr["regime_results"]
        if not isinstance(regime_results, dict):
            continue

        for scan_label, grid_res in regime_results.items():
            yaml_key = regime_label_to_yaml_key(str(scan_label))
            if not yaml_key:
                continue
            if not isinstance(grid_res, dict):
                continue
            params = grid_res.get("best_params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                params = {}

            routing.setdefault(asset, {}).setdefault(freq, {})[yaml_key] = {
                "models": [
                    {
                        "name": strat,
                        "weight": 1.0,
                        "params": copy.deepcopy(params),
                    }
                ]
            }
            n += 1
    return n


def write_strategies_draft_yaml(
    tuning_results: list[dict[str, Any]],
    *,
    base_path: Optional[Path | str] = None,
    out_path: Optional[Path | str] = None,
    result_dir: str = "result",
    rank_slot: int = 1,
) -> Optional[str]:
    """
    Load base strategies.yaml, deep-copy, apply per-regime tune patches, write draft file.

    Non-regime grid search (single best_params for whole run) is not applied automatically —
    regimes in YAML are ambiguous; merge those by hand from JSON.

    Returns written path, or None if nothing written (no yaml, no patches, empty tuning).
    """
    try:
        import yaml
    except Exception:
        print("[Draft] PyYAML not available; skip strategies draft export.")
        return None

    if not tuning_results:
        print("[Draft] No tuning_results; skip strategies draft export.")
        return None

    base = Path(base_path) if base_path else _default_strategies_yaml_path()
    if not base.is_file():
        print(f"[Draft] Base file not found: {base}; skip.")
        return None

    with open(base, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        print("[Draft] Invalid YAML root; skip.")
        return None

    patched = copy.deepcopy(doc)
    n = apply_tune_regimes_to_routing(patched, tuning_results, rank_slot=rank_slot)
    if n == 0:
        print(
            "[Draft] No use_regimes tuning entries to merge (need --tune-best --tune-regimes); skip file."
        )
        return None

    patched["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )

    out: Path
    if out_path:
        out = Path(out_path)
    else:
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path(result_dir) / f"strategies_draft_{stamp}.yaml"

    header = (
        "# DRAFT — generated after run_strategy_scan + tune; review before replacing config/strategies.yaml.\n"
        f"# Patched {n} regime cell(s) from tune-regimes results (rank_slot={rank_slot}).\n"
        f"# Base: {base}\n"
        "# Non-regime-only tuning is not auto-merged (see JSON tuning_results).\n\n"
    )

    body = yaml.safe_dump(
        patched,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(body)

    print(f"\n[Draft] Wrote strategies draft: {out}")
    return str(out)


def write_strategies_draft_from_regime_winners(
    per_regime_winners: dict[str, Any],
    *,
    base_path: Optional[Path | str] = None,
    out_path: Optional[Path | str] = None,
    result_dir: str = "result",
) -> Optional[str]:
    """
    Write draft YAML from ``per_regime_winners`` (best strategy + params per regime cell).

    Structure: ``winners[asset][frequency][regime]`` → ``{strategy, best_score, best_params}``.
    Regime labels are mapped with ``regime_label_to_yaml_key`` (pattern bullish→bull, …;
    production labels already match ``strategies.yaml``).
    """
    try:
        import yaml
    except Exception:
        print("[Draft] PyYAML not available; skip strategies draft export.")
        return None

    if not per_regime_winners:
        print("[Draft] No per_regime_winners; skip.")
        return None

    base = Path(base_path) if base_path else _default_strategies_yaml_path()
    if not base.is_file():
        print(f"[Draft] Base file not found: {base}; skip.")
        return None

    with open(base, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        print("[Draft] Invalid YAML root; skip.")
        return None

    patched = copy.deepcopy(doc)
    routing = patched.setdefault("routing", {})
    n = 0
    for asset, fm in per_regime_winners.items():
        if not isinstance(fm, dict):
            continue
        for freq, regimes in fm.items():
            if not isinstance(regimes, dict):
                continue
            for regime, detail in regimes.items():
                if not isinstance(detail, dict):
                    continue
                yaml_key = regime_label_to_yaml_key(str(regime))
                if not yaml_key:
                    print(
                        f"[Draft] Skip unknown regime label {regime!r} for "
                        f"{asset}/{freq} (not in YAML taxonomy)."
                    )
                    continue
                name = detail.get("strategy")
                if not name:
                    continue
                params = detail.get("best_params") or {}
                if not isinstance(params, dict):
                    params = {}
                routing.setdefault(asset, {}).setdefault(freq, {})[yaml_key] = {
                    "models": [
                        {"name": str(name), "weight": 1.0, "params": copy.deepcopy(params)}
                    ]
                }
                n += 1

    if n == 0:
        print("[Draft] No regime cells in per_regime_winners; skip file.")
        return None

    patched["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )

    out: Path
    if out_path:
        out = Path(out_path)
    else:
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path(result_dir) / f"strategies_draft_regime_winners_{stamp}.yaml"

    header = (
        "# DRAFT — built from per_regime_winners (best strategy per regime after comparing tuned scores).\n"
        f"# Patched {n} regime cell(s).\n"
        f"# Base: {base}\n\n"
    )
    body = yaml.safe_dump(
        patched,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(body)
    print(f"\n[Draft] Wrote regime-winner strategies draft: {out}")
    return str(out)
