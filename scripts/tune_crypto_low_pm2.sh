#!/usr/bin/env bash
# Tune BTC/ETH/SOL/XAU low: production detect_regime (crypto → _detect_crypto_low on 5m).
# Usage: pm2 start scripts/tune_crypto_low_pm2.sh --name tune-crypto-low --interpreter bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
TS="$(date -u +%Y%m%d_%H%M%S)"
exec uv run python -m synth.miner.run_strategy_scan \
  --assets BTC ETH SOL XAU \
  --frequencies low \
  --tune-best \
  --tune-regimes \
  --write-strategies-draft \
  --result-dir "result/tune_crypto_low_pm2_${TS}"
