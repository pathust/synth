# Developer Onboarding Guide

Tài liệu này dành cho developer mới tham gia project Synth, tập trung vào 4 luồng chính: phát triển strategy mới, tuning, backtest, và visualize.

## 1) Cấu trúc project, module, design pattern, coding convention

### 1.1 Cấu trúc tổng quan

```text
synth/
├── neurons/                     # Entry point miner / validator
├── synth/
│   ├── base/                    # Base neuron, networking helpers
│   ├── db/                      # SQLAlchemy models
│   ├── miner/                   # Strategy, backtest, tuning, deploy, viz
│   ├── validator/               # Scoring pipeline và validator runtime
│   ├── protocol.py              # Synapse schema
│   └── simulation_input.py      # Input schema cho mô phỏng
├── scripts/                     # Script vận hành/backtest tiện ích
├── tests/                       # Test chính
├── refactor_tests/              # Test cho kiến trúc mới
└── docs/                        # Spec và tài liệu kỹ thuật
```

### 1.2 Entry points quan trọng

- Miner runtime: [miner.py](file:///Users/taiphan/Documents/synth/neurons/miner.py)
- Validator runtime: [validator.py](file:///Users/taiphan/Documents/synth/neurons/validator.py)
- Entry mô phỏng thống nhất: [entry.py](file:///Users/taiphan/Documents/synth/synth/miner/entry.py)
- Strategy scan CLI: [run_strategy_scan.py](file:///Users/taiphan/Documents/synth/synth/miner/run_strategy_scan.py)

### 1.3 Module lõi trong `synth/miner`

- `strategies/`
  - Base class + contract: [base.py](file:///Users/taiphan/Documents/synth/synth/miner/strategies/base.py)
  - Registry auto-discovery: [registry.py](file:///Users/taiphan/Documents/synth/synth/miner/strategies/registry.py)
- `config/`
  - Runtime config store từ YAML: [strategy_store.py](file:///Users/taiphan/Documents/synth/synth/miner/config/strategy_store.py)
  - Production mapping hiện hành: [strategies.yaml](file:///Users/taiphan/Documents/synth/synth/miner/config/strategies.yaml)
- `backtest/`
  - Engine orchestration: [engine.py](file:///Users/taiphan/Documents/synth/synth/miner/backtest/engine.py)
  - Runner benchmark: [runner.py](file:///Users/taiphan/Documents/synth/synth/miner/backtest/runner.py)
  - GridSearch tuner: [tuner.py](file:///Users/taiphan/Documents/synth/synth/miner/backtest/tuner.py)
  - Report exporter/ranking: [report.py](file:///Users/taiphan/Documents/synth/synth/miner/backtest/report.py)
- `deploy/`
  - Export kết quả tuning/backtest thành YAML: [exporter.py](file:///Users/taiphan/Documents/synth/synth/miner/deploy/exporter.py)
  - Validate config trước apply: [applier.py](file:///Users/taiphan/Documents/synth/synth/miner/deploy/applier.py)
- `viz/`
  - HTML backtest report: [backtest_report.py](file:///Users/taiphan/Documents/synth/synth/miner/viz/backtest_report.py)

### 1.4 Design patterns đang dùng

- Strategy Pattern: mỗi model là một `BaseStrategy` với `simulate(...)`.
- Registry Pattern: `StrategyRegistry` quản lý discovery/lookup theo `asset` và `frequency`.
- Config-Driven Routing: route chiến thuật qua `strategies.yaml` thay vì hardcode.
- Fallback Chain: nếu strategy chính lỗi, `entry.py` chạy chain fallback.
- Compatibility Layer: module cũ (`entry_new`, `backtest/framework`) delegate về API mới để tránh breaking runtime.

### 1.5 Coding conventions

- Formatting/lint:
  - [pyproject.toml](file:///Users/taiphan/Documents/synth/pyproject.toml)
  - [.pre-commit-config.yaml](file:///Users/taiphan/Documents/synth/.pre-commit-config.yaml)
- Python env và test:
  - Dùng `conda run -n synth ...`
- Test policy:
  - `tests/conftest.py` đã chuyển DB fixture sang on-demand; chỉ test có `db_engine` mới cần Docker.

## 2) Quy trình 6 bước để viết strategy mới

### Bước 1: Định nghĩa concept và hypothesis

- Xác định market regime mục tiêu: trending / mean-reverting / high-vol.
- Xác định frequency hỗ trợ: `high` (1h) hoặc `low` (24h).
- Đặt KPI: CRPS trung bình, drawdown penalty, stability qua nhiều mốc thời gian.

### Bước 2: Thiết kế interface strategy

- Strategy phải implement đúng chữ ký trong `BaseStrategy.simulate(...)`.
- Trả về `np.ndarray` shape `(n_sims, steps+1)`.
- Hỗ trợ `seed` để reproducible.

Template: [strategy_template.py](file:///Users/taiphan/Documents/synth/docs/templates/strategy_template.py)

### Bước 3: Implementation + parameter surface

- Set metadata rõ ràng:
  - `name`, `supported_assets`, `supported_frequencies`
  - `default_params`, `param_grid`
- Tách rõ:
  - phần estimate từ historical data
  - phần sinh paths
  - phần guard cho edge cases (NaN, giá <= 0, overflow)

### Bước 4: Unit test requirements

Bắt buộc có tối thiểu:

- `test_shape`: output đúng `(n_sims, steps+1)`
- `test_reproducibility`: cùng `seed` cho output giống nhau
- `test_no_nan_inf`: không có NaN/Inf
- `test_positive_prices`: giá đầu ra > 0
- `test_short_history_guard`: input ít dữ liệu vẫn xử lý an toàn

Template: [strategy_test_template.py](file:///Users/taiphan/Documents/synth/docs/templates/strategy_test_template.py)

### Bước 5: Risk management integration

- Áp dụng guard trong strategy:
  - clamp volatility
  - clamp jump size
  - variance floor
- Áp dụng risk overlay ở post-processing:
  - max leverage cap
  - stop-loss trigger simulation
  - position sizing theo volatility bucket

### Bước 6: Đăng ký và đưa vào runtime routing

- Đảm bảo strategy được auto-discover bởi registry.
- Chạy scan/tuning để lấy ranking thực nghiệm.
- Cập nhật `strategies.yaml` qua deploy exporter (không sửa tay production trừ khi cần hotfix).

## 3) Tuning framework: Grid Search, Walk-Forward, Cross-Validation

### 3.1 Grid Search chuẩn hiện tại

- Dùng `GridSearchTuner` trong [tuner.py](file:///Users/taiphan/Documents/synth/synth/miner/backtest/tuner.py).
- Nguồn `param_grid`:
  - từ strategy (`get_param_grid`) hoặc
  - override từ config.

Ví dụ scan + tune:

```bash
PYTHONPATH=. conda run -n synth python -m synth.miner.run_strategy_scan \
  --assets BTC \
  --frequencies high low \
  --num-runs 5 \
  --num-sims 50 \
  --tune-best \
  --result-dir result/btc_scan
```

### 3.2 Walk-Forward Optimization (WFO)

Khuyến nghị chuẩn hóa theo rolling window:

- Train window: `T_train` ngày
- Validate window: `T_val` ngày
- Roll step: `T_step` ngày
- Mỗi fold:
  - tune params trên train window
  - freeze params
  - backtest trên validate window
- Tổng hợp:
  - mean CRPS
  - std CRPS
  - tail risk stats

### 3.3 Cross-validation kỹ thuật cho time-series

- Anchored CV
  - Train set mở rộng dần, validate trên block kế tiếp.
- Rolling CV
  - Cửa sổ train cố định trượt theo thời gian.
- Purged CV
  - Loại bỏ vùng rò rỉ thông tin giữa train/validate quanh boundary.

### 3.4 Performance criteria

Tối thiểu đánh giá:

- CRPS mean, median, p90/p95
- Stability score: `std(CRPS)` theo folds
- Success rate: số run thành công / tổng run
- Runtime latency per simulation batch
- Failure reason taxonomy (OOM, convergence fail, invalid output)

Template config tuning: [tuning_config_template.yaml](file:///Users/taiphan/Documents/synth/docs/templates/tuning_config_template.yaml)

## 4) Pipeline backtest hoàn chỉnh

### 4.1 Data preprocessing

- Load historical prices từ DB qua `DataHandler`.
- Filter dữ liệu trước `start_time` để tránh leakage.
- Chuẩn hóa resolution theo frequency:
  - high: 1m
  - low: 5m
- Validate continuity và dữ liệu thiếu.

### 4.2 Execution simulation

- Gọi `strategy.simulate(...)` với seed xác định.
- Convert output thành prediction format chuẩn validator.
- Validate format qua `validate_responses`.

### 4.3 Transaction cost modeling

Khuyến nghị chuẩn trong backtest:

- Fee model:
  - `fee = notional * fee_bps / 10_000`
- Spread model:
  - `spread_cost = notional * spread_bps / 10_000`
- Total cost:
  - `total_cost = fee + spread_cost`

### 4.4 Slippage calculation

Khuyến nghị mô hình:

- Linear impact:
  - `slippage_bps = alpha * participation_rate`
- Volatility-adjusted impact:
  - `slippage_bps = alpha * participation_rate * (1 + beta * realized_vol)`
- Execution price:
  - Buy: `px_exec = px_mid * (1 + slippage_bps/10_000)`
  - Sell: `px_exec = px_mid * (1 - slippage_bps/10_000)`

### 4.5 Post-simulation scoring

- Primary metric: CRPS.
- Secondary metrics:
  - RMSE, MAE, directional accuracy, tail error.
- Tính aggregate theo:
  - per run
  - per strategy
  - per frequency

Workflow demo script: [backtest_workflow_demo.sh](file:///Users/taiphan/Documents/synth/docs/templates/backtest_workflow_demo.sh)

## 5) Visualization dashboard requirements

### 5.1 10+ chart bắt buộc

Dashboard phải có tối thiểu:

1. Equity curve (PnL cumulative)
2. Daily/periodic returns histogram
3. Drawdown curve
4. Rolling Sharpe ratio
5. Rolling Sortino ratio
6. Win rate theo tháng/tuần
7. Profit factor
8. Exposure / position utilization
9. Turnover chart
10. Slippage cost breakdown
11. Transaction cost breakdown (fee/spread/impact)
12. CRPS distribution (boxplot/hist)
13. Strategy ranking heatmap theo asset × frequency
14. Stability chart (score mean/std theo folds)

### 5.2 Interactive features bắt buộc

- Filter theo:
  - asset
  - strategy
  - frequency
  - date range
- Hover tooltip đầy đủ metadata.
- Drill-down từ aggregate -> từng run.
- Export PNG/CSV/JSON.
- So sánh side-by-side nhiều strategy cùng lúc.

### 5.3 Output format

- HTML report cho stakeholder.
- JSON raw metrics cho phân tích lại.
- PNG cho chart chính.

## 6) Checklist validation trước khi merge (20+ mục)

### 6.1 Code correctness

1. Strategy implement đủ method `simulate`.
2. Output shape đúng ở mọi branch logic.
3. Không có NaN/Inf trong paths.
4. Seed reproducibility pass.
5. Guard khi dữ liệu lịch sử ngắn.
6. Guard khi volatility estimate bất thường.
7. Guard khi tham số đầu vào ngoài range.
8. Không dùng global mutable state.

### 6.2 Data integrity

9. Không có data leakage qua `start_time`.
10. Timeframe mapping đúng (`high->1m`, `low->5m`).
11. Missing data được xử lý rõ ràng.
12. Timestamps parse thống nhất timezone.

### 6.3 Backtest quality

13. Chạy benchmark tối thiểu `num_runs >= 3`.
14. Chạy `num_sims=50` cho quick test và profile runtime.
15. So sánh với baseline strategy.
16. Có báo cáo ranking theo frequency.
17. Có lưu artifact JSON kết quả.

### 6.4 Tuning quality

18. Param grid có ý nghĩa thực nghiệm.
19. Có tiêu chí chọn best params rõ ràng.
20. Có kiểm tra overfitting (WFO/CV).
21. Báo cáo score distribution, không chỉ average.

### 6.5 Risk & deployment

22. Kiểm tra fallback behavior khi strategy fail.
23. Kiểm tra config `strategies.yaml` pass validator.
24. Apply config qua deploy applier.
25. Backup config cũ trước khi ghi đè.

### 6.6 Testing & docs

26. Unit tests pass cho strategy mới.
27. `refactor_tests` pass.
28. Full pytest pass/skip hợp lệ.
29. Cập nhật tài liệu kiến trúc nếu đổi contract.
30. Cập nhật changelog/refactor status.

## Walkthrough demo toàn bộ workflow

### Bước A: Tạo strategy mới

1. Copy template: [strategy_template.py](file:///Users/taiphan/Documents/synth/docs/templates/strategy_template.py)
2. Đặt file vào `synth/miner/strategies/<your_strategy>.py`
3. Cập nhật tên strategy và param grid.

### Bước B: Viết unit test

1. Copy template: [strategy_test_template.py](file:///Users/taiphan/Documents/synth/docs/templates/strategy_test_template.py)
2. Đặt vào `refactor_tests/test_<your_strategy>.py`
3. Chạy:

```bash
conda run -n synth python -m pytest refactor_tests -q
```

### Bước C: Scan + tuning

```bash
PYTHONPATH=. conda run -n synth python -m synth.miner.run_strategy_scan \
  --assets BTC \
  --frequencies high low \
  --num-runs 5 \
  --num-sims 50 \
  --seed 42 \
  --window-days 30 \
  --tune-best \
  --result-dir result/dev_scan
```

### Bước D: Export + apply config

```bash
PYTHONPATH=. conda run -n synth python -c "
import json
from synth.miner.deploy.exporter import export_best_config
from synth.miner.deploy.applier import apply_to_miner
scan = json.load(open('result/dev_scan/strategy_scan_YYYYMMDD_HHMMSS.json','r',encoding='utf-8'))['scan_results']
export_best_config(scan_results=scan, top_n=3, output_path='synth/miner/config/strategies.yaml', backup=True, min_successful_runs=2)
print(apply_to_miner(config_path='synth/miner/config/strategies.yaml'))
"
```

### Bước E: Visualize

```bash
PYTHONPATH=. conda run -n synth python scripts/btc_backtest_visualize.py
```

### Bước F: Final validation trước merge

```bash
conda run -n synth python -m pytest -q
```

## Deliverables đi kèm

- Onboarding guide: file này.
- Code templates:
  - [strategy_template.py](file:///Users/taiphan/Documents/synth/docs/templates/strategy_template.py)
  - [strategy_test_template.py](file:///Users/taiphan/Documents/synth/docs/templates/strategy_test_template.py)
- Config template:
  - [tuning_config_template.yaml](file:///Users/taiphan/Documents/synth/docs/templates/tuning_config_template.yaml)
- Workflow demo script:
  - [backtest_workflow_demo.sh](file:///Users/taiphan/Documents/synth/docs/templates/backtest_workflow_demo.sh)
