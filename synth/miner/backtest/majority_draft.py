"""
majority_draft.py — Majority-vote aggregator for tuned regime-winner drafts.

Reads multiple ``strategies_draft_regime_winners_*.yaml`` files and computes the
most frequently selected top-1 model per (asset, frequency, regime). Outputs a
single draft YAML with those majority picks.

By request:
  - High uses only drafts under result/tune_high/
  - Low uses only drafts under result/tune_low/
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, DefaultDict

import yaml


@dataclass(frozen=True)
class CellKey:
    asset: str
    frequency: str
    regime: str


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _iter_draft_files(dir_path: str) -> list[str]:
    if not dir_path:
        return []
    if not os.path.isdir(dir_path):
        return []
    out: list[str] = []
    for name in os.listdir(dir_path):
        if name.startswith("strategies_draft_regime_winners_") and name.endswith(".yaml"):
            out.append(os.path.join(dir_path, name))
    return sorted(out)


def _freeze_obj(x: Any) -> Any:
    """
    Convert nested dict/list/scalars into a hashable, deterministic structure so we can
    count "same params" across YAML files.
    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return tuple((str(k), _freeze_obj(v)) for k, v in sorted(x.items(), key=lambda kv: str(kv[0])))
    if isinstance(x, list):
        return tuple(_freeze_obj(v) for v in x)
    # Fallback: stringify (covers numpy scalars etc.)
    return str(x)


def _top1_model(cell: Any) -> tuple[str, dict[str, Any]] | None:
    """
    Cell shape in drafts:
      { models: [ {name: ..., weight: 1.0, params: {...}}, ... ] }
    We take only the first model (top-1) and its params.
    """
    if not isinstance(cell, dict):
        return None
    models = cell.get("models")
    if not isinstance(models, list) or not models:
        return None
    m0 = models[0]
    if not isinstance(m0, dict):
        return None
    name = m0.get("name")
    if not name:
        return None
    params = m0.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    return str(name), dict(params)


def _deep_get_routing(doc: dict[str, Any]) -> dict[str, Any]:
    routing = doc.get("routing") or {}
    if not isinstance(routing, dict):
        return {}
    return routing


def _collect_cells(
    draft_paths: list[str],
    *,
    frequency_filter: str,
) -> tuple[dict[CellKey, list[tuple[int, str, dict[str, Any]]]], dict[str, Any] | None]:
    """
    Return all observed top-1 picks per cell.

    Each entry is (file_order, strategy_name, params_dict) where file_order increases
    with draft_paths ordering (so the last is "newest" if paths are sorted ascending).
    """
    cells: dict[CellKey, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    first_doc: dict[str, Any] | None = None

    for i, p in enumerate(draft_paths):
        doc = _load_yaml(p)
        if first_doc is None:
            first_doc = doc
        routing = _deep_get_routing(doc)
        for asset, asset_block in routing.items():
            if not isinstance(asset_block, dict):
                continue
            freq_block = asset_block.get(frequency_filter)
            if not isinstance(freq_block, dict):
                continue
            for regime, cell in freq_block.items():
                top = _top1_model(cell)
                if not top:
                    continue
                name, params = top
                cells[CellKey(str(asset), frequency_filter, str(regime))].append((i, name, params))

    return dict(cells), first_doc


def _majority_name(picks: list[tuple[int, str, dict[str, Any]]]) -> tuple[str, int]:
    c = Counter([name for _i, name, _params in picks if name])
    if not c:
        raise ValueError("no names to vote on")
    best_n = max(c.values())
    best_names = sorted([k for k, v in c.items() if v == best_n])
    return best_names[0], best_n


def _pick_params_for_name(
    picks: list[tuple[int, str, dict[str, Any]]],
    *,
    name: str,
) -> dict[str, Any]:
    """
    Rule:
      - Filter to only picks with the winning strategy name.
      - Majority vote on params fingerprint inside that name.
      - If no clear "same params" majority (i.e. winner count == 1 and >1 variants),
        or ties, pick params from the newest draft (largest file_order) among that name.
    """
    same = [(i, params) for i, n, params in picks if n == name]
    if not same:
        return {}

    fp_counts: Counter[Any] = Counter(_freeze_obj(p) for _i, p in same)
    if len(fp_counts) == 1:
        # Everyone agrees
        return same[-1][1]  # newest is fine

    best = max(fp_counts.values())
    best_fps = [fp for fp, v in fp_counts.items() if v == best]
    # If there's a unique winning fingerprint with count>1, use that.
    if len(best_fps) == 1 and best > 1:
        fp = best_fps[0]
        # pick newest occurrence of that fp (stable)
        for i, params in reversed(same):
            if _freeze_obj(params) == fp:
                return params
        return same[-1][1]

    # Otherwise: "không trùng" or tie → pick newest params for that strategy name.
    return max(same, key=lambda t: t[0])[1]


def _build_majority_draft(
    *,
    base_doc: dict[str, Any],
    majority_by_cell: dict[CellKey, tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Keep top-level metadata if present
    for k in ["version", "fallback_chain"]:
        if k in base_doc:
            out[k] = base_doc[k]
    out["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["source"] = {
        "method": "majority_vote_top1_name_and_params",
    }

    routing_out: dict[str, Any] = {}
    base_routing = _deep_get_routing(base_doc)

    # Use base_doc routing skeleton so we keep only known assets/regs from base
    for asset, asset_block in base_routing.items():
        if not isinstance(asset_block, dict):
            continue
        asset_out: dict[str, Any] = {}
        for freq in ["high", "low"]:
            freq_block = asset_block.get(freq)
            if not isinstance(freq_block, dict):
                continue
            freq_out: dict[str, Any] = {}
            for regime in freq_block.keys():
                key = CellKey(str(asset), str(freq), str(regime))
                if key not in majority_by_cell:
                    continue
                name, params = majority_by_cell[key]
                freq_out[str(regime)] = {
                    "models": [
                        {
                            "name": name,
                            "weight": 1.0,
                            "params": params or {},
                        }
                    ]
                }
            if freq_out:
                asset_out[freq] = freq_out
        if asset_out:
            routing_out[str(asset)] = asset_out

    out["routing"] = routing_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Majority-vote aggregator over strategies_draft_regime_winners_*.yaml"
    )
    ap.add_argument("--tune-high-dir", default="result/tune_high")
    ap.add_argument("--tune-low-dir", default="result/tune_low")
    ap.add_argument(
        "--base-yaml",
        default="synth/miner/config/strategies.yaml",
        help="Base YAML used as routing skeleton (assets/frequencies/regimes).",
    )
    ap.add_argument(
        "--out",
        default="result/strategies_draft_majority_vote.yaml",
        help="Output path for majority-vote draft YAML.",
    )
    args = ap.parse_args()

    high_paths = _iter_draft_files(str(args.tune_high_dir))
    low_paths = _iter_draft_files(str(args.tune_low_dir))
    if not high_paths and not low_paths:
        raise SystemExit("No draft YAML files found under tune_high_dir/tune_low_dir")

    base_doc = _load_yaml(os.path.abspath(str(args.base_yaml)))

    cells_high, _doc0_h = _collect_cells(high_paths, frequency_filter="high")
    cells_low, _doc0_l = _collect_cells(low_paths, frequency_filter="low")

    majority_by_cell: dict[CellKey, tuple[str, dict[str, Any]]] = {}
    summary_lines: list[str] = []

    # High: only from tune_high (frequency=high)
    for key, picks in sorted(cells_high.items(), key=lambda kv: (kv[0].asset, kv[0].frequency, kv[0].regime)):
        winner_name, n = _majority_name(picks)
        params = _pick_params_for_name(picks, name=winner_name)
        majority_by_cell[key] = (winner_name, params)
        summary_lines.append(f"{key.asset} {key.frequency} {key.regime}: {winner_name} ({n}/{len(picks)})")

    # Low: only from tune_low (frequency=low)
    for key, picks in sorted(cells_low.items(), key=lambda kv: (kv[0].asset, kv[0].frequency, kv[0].regime)):
        winner_name, n = _majority_name(picks)
        params = _pick_params_for_name(picks, name=winner_name)
        majority_by_cell[key] = (winner_name, params)
        summary_lines.append(f"{key.asset} {key.frequency} {key.regime}: {winner_name} ({n}/{len(picks)})")

    out_doc = _build_majority_draft(base_doc=base_doc, majority_by_cell=majority_by_cell)

    os.makedirs(os.path.dirname(os.path.abspath(str(args.out))) or ".", exist_ok=True)
    with open(str(args.out), "w", encoding="utf-8") as f:
        f.write("# DRAFT — majority vote across tuned drafts\n")
        f.write(f"# tune_high_dir: {os.path.abspath(str(args.tune_high_dir))} ({len(high_paths)} files)\n")
        f.write(f"# tune_low_dir:  {os.path.abspath(str(args.tune_low_dir))} ({len(low_paths)} files)\n")
        f.write("# Rule: pick most frequent top-1 model per (asset, frequency, regime)\n")
        f.write("#\n")
        yaml.safe_dump(out_doc, f, sort_keys=False, allow_unicode=True)

    print(f"[majority_draft] wrote: {os.path.abspath(str(args.out))}")
    print("[majority_draft] summary (winner counts):")
    for line in summary_lines:
        print("  - " + line)


if __name__ == "__main__":
    main()

