# Regime Prediction & Backtest Engine

Tài liệu này mô tả luồng backtest/tuning theo taxonomy:

`[asset]/[type]/[regime]/[strategy]`

Ví dụ:
- `btc/high/bearish/garch_v4`
- `xau/spot/mean_reverting/garch_v4_1`
- `aaplx/session/market_open/arima_equity`

## 1. Thành phần chính

- `synth/miner/backtest/regime_engine.py`
  - `MarketTaxonomyRouter`: cắt DataFrame lịch sử thành từng case theo if-else.
  - `PredictionBacktestEngine`: tune + validation/test cho từng case.
  - `RegimeEngineConfig`: cấu hình split, giới hạn sample, số fold.

- `synth/miner/backtest/tuner.py`
  - `GridSearchTuner.run(...)` hỗ trợ `dates=` để tune theo tập thời gian cụ thể.

## 2. Quy trình

1. **Slice dữ liệu** theo taxonomy:
   - Crypto: phân loại `high/low` từ rolling volatility + volume.
   - Crypto high: regime `bullish/bearish/neutral`; crypto low: `unknown`.
   - XAU: regime `mean_reverting/trending` theo ER.
   - Equity: regime `market_open/overnight` (và `earnings` nếu có lịch).

2. **Split time-series** (single hoặc walk-forward):
   - Train -> Tuning params
   - Validation -> Chọn strategy
   - Test -> Đánh giá cuối

3. **Tuning/Evaluation per case**:
   - Tune từng strategy trên train dates.
   - So sánh theo validation score.
   - Chốt strategy tốt nhất, đo test score.

4. **Export**:
   - JSON report
   - Taxonomy YAML (`asset -> type -> regime`)
   - Runtime YAML (`strategies.yaml` schema)

## 3. Chạy từ CSV

```bash
PYTHONPATH=. conda run -n synth python -m synth.miner.run_regime_backtest \
  --input-csv /path/to/history.csv \
  --output-dir result/regime_backtest \
  --timestamp-col timestamp \
  --asset-col asset \
  --price-col close \
  --volume-col volume \
  --split-mode walk_forward \
  --walk-forward-folds 3
```

Output:
- `regime_report_*.json`
- `regime_taxonomy_*.yaml`
- `strategies_runtime_*.yaml`

