#!/usr/bin/env bash
set -euo pipefail

cd /Users/taiphan/Documents/synth

PYTHONPATH=. conda run -n synth python -m pytest refactor_tests -q

PYTHONPATH=. conda run -n synth python -m synth.miner.run_strategy_scan \
  --assets BTC \
  --frequencies high low \
  --num-runs 5 \
  --num-sims 50 \
  --seed 42 \
  --window-days 30 \
  --tune-best \
  --result-dir result/dev_scan

PYTHONPATH=. conda run -n synth python -c "
import glob, json
from synth.miner.deploy.exporter import export_best_config
from synth.miner.deploy.applier import apply_to_miner
latest = sorted(glob.glob('result/dev_scan/strategy_scan_*.json'))[-1]
scan = json.load(open(latest, 'r', encoding='utf-8'))['scan_results']
export_best_config(
    scan_results=scan,
    top_n=3,
    output_path='synth/miner/config/strategies.yaml',
    backup=True,
    min_successful_runs=2
)
print(apply_to_miner(config_path='synth/miner/config/strategies.yaml'))
"

PYTHONPATH=. conda run -n synth python scripts/btc_backtest_visualize.py

conda run -n synth python -m pytest -q
