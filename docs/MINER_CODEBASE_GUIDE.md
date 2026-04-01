# Synth Miner Codebase Guide (Deep Dive)

Tài liệu này tập trung riêng vào `synth/miner`, dùng để:

- đọc hiểu nhanh codebase hiện tại,
- biết sửa đúng chỗ khi implement/chỉnh sửa,
- tránh đụng vào các điểm dễ gây regression runtime.

## 1. Bức tranh tổng thể `synth/miner`

`synth/miner` hiện vận hành theo trục:

1. **request runtime** (miner nhận prompt từ validator),
2. **strategy simulation** (single strategy hoặc ensemble),
3. **backtest + tuning** (đánh giá hiệu năng chiến thuật),
4. **deploy config** (`strategies.yaml`),
5. **visualization/reporting**.

Luồng runtime chính:

`neurons/miner.py` → `synth/miner/entry.py` → strategy/ensemble/fallback → formatted predictions.

## 2. Module map chi tiết

## 2.1 Runtime orchestration

- `entry.py`
  - điểm vào chuẩn của miner cho generate simulation.
  - chứa:
    - strategy name → simulate function map,
    - gọi ensemble phase trước,
    - nếu fail thì chạy fallback chain tuần tự,
    - validate format output trước khi trả.
- `entry_new.py`
  - compatibility wrapper, không phải đường chính.
- `run.py`
  - local benchmark/smoke script.

## 2.2 Strategy system

- `strategies/base.py`
  - contract bắt buộc của mọi strategy (`simulate`).
  - định nghĩa metadata (`name`, `supported_assets`, `supported_frequencies`, `param_grid`, `default_params`).
- `strategies/registry.py`
  - auto-discovery strategy classes trong package.
  - filter theo asset/frequency.
- `strategies/*.py`
  - từng implementation cụ thể (garch, egarch, jump, regime, router...).

## 2.3 Data & simulation pipeline

- `my_simulation.py`
  - core helper để tạo price paths từ strategy function.
  - cầu nối giữa request config và data loader.
- `data/dataloader.py`
  - lấy dữ liệu lịch sử và đảm bảo OOS boundary.
- `pipeline/`
  - `base.py`: abstraction `DataProvider`.
  - `loader.py`: orchestration nhiều provider.
  - `providers/`: MySQL/DuckDB/Pyth provider.
- `data_handler.py`
  - thao tác dữ liệu lịch sử từ DB/file, fetch/backfill, convert timeframe.

## 2.4 Config & routing

- `config/strategy_store.py`
  - đọc `strategies.yaml`, cache theo mtime, reload khi thay đổi.
  - fallback về mapping Python nếu YAML lỗi/thiếu.
- `config/strategies.yaml`
  - source-of-truth runtime routing strategy.
- `config/asset_strategy_config.py`
  - fallback mapping cũ để compatibility.

## 2.5 Ensemble

- `ensemble/builder.py`
  - build weighted ensemble từ list strategy configs.
- `ensemble/trimmer.py`
  - trimming/pooling logic cho outputs.

## 2.6 Backtest/Tuning

- `backtest/runner.py`
  - run single/benchmark/scan-all.
- `backtest/tuner.py`
  - GridSearch over param grid.
- `backtest/engine.py`
  - orchestration cấp cao cho run/tune/export/visualize.
- `backtest/metrics.py`
  - metric functions.
- `backtest/report.py`
  - ranking + export JSON artifacts.
- `backtest/duel.py`
  - đối đầu 2 strategies.

## 2.7 Deploy & Visualization

- `deploy/exporter.py`
  - chuyển kết quả backtest/tuning thành `strategies.yaml` qua atomic swap.
- `deploy/applier.py`
  - validate YAML schema + strategy existence trước apply.
- `viz/strategy_compare.py`, `viz/backtest_report.py`
  - chart PNG + HTML report.

## 2.8 Legacy zone

- `_legacy/`
  - chỉ tham khảo.
  - không dùng làm source chính cho runtime mới.

## 3. Design rules quan trọng khi chỉnh sửa

1. **Không bypass `entry.py`** cho production path.
2. **Không hardcode routing** trong code runtime; cập nhật qua `strategies.yaml` hoặc deploy exporter.
3. **Giữ OOS boundary** trong dataloader, không leak dữ liệu tương lai.
4. **Mọi strategy mới phải tương thích `BaseStrategy.simulate` signature**.
5. **Không bỏ validate output** trước khi trả về validator.
6. **Ưu tiên sửa module mới (`backtest/engine.py`, `strategy_store.py`) thay vì legacy.**

## 4. Playbook implement/chỉnh sửa theo nhu cầu

## 4.1 Thêm strategy mới

Bước chuẩn:

1. tạo file `synth/miner/strategies/<new_strategy>.py`,
2. subclass `BaseStrategy`,
3. implement `simulate(...)`,
4. set metadata + `param_grid`,
5. thêm test,
6. chạy scan và so sánh ranking.

Checklist bắt buộc:

- output shape `(n_sims, steps+1)`,
- seed reproducible,
- không NaN/Inf,
- guard input thiếu data.

## 4.2 Chỉnh routing strategy theo kết quả backtest

Khuyến nghị flow:

1. chạy `run_strategy_scan.py`,
2. export top strategies bằng `deploy/exporter.py`,
3. validate bằng `deploy/applier.py`,
4. apply vào `config/strategies.yaml`.

Không nên chỉnh tay file YAML trong production nếu chưa validate.

## 4.3 Chỉnh logic fallback runtime

Sửa trong `entry.py`:

- giữ thứ tự phase:
  - phase 1: ensemble attempt,
  - phase 2: fallback chain.
- mọi thay đổi phải giữ format validation.

Regression test cần chạy:

- test fallback khi strategy chính fail,
- test output format pass validator.

## 4.4 Chỉnh data source/provider

Sửa trong `pipeline/providers/*` + `pipeline/loader.py`.

Nguyên tắc:

- provider mới phải implement `DataProvider.fetch`,
- giữ backward compatibility cho MySQL path hiện có,
- validate data continuity trước khi feed strategy.

## 4.5 Chỉnh backtest metric/tuning

- Metric mới:
  - thêm vào `backtest/metrics.py`,
  - đăng ký vào metric map runner.
- Tuning:
  - update `param_grid` ở strategy,
  - tune bằng `GridSearchTuner`,
  - xác nhận bằng benchmark nhiều run.

## 5. Safe-edit matrix (sửa file nào cho mục tiêu nào)

- **Thêm strategy** → `strategies/*.py`, `strategies/base.py` (tham chiếu), tests.
- **Thay route theo asset/frequency** → `config/strategies.yaml`, `deploy/exporter.py`, `deploy/applier.py`.
- **Đổi runtime fallback** → `entry.py`.
- **Đổi data ingest/fetch** → `fetch_daemon.py`, `data_handler.py`, `pipeline/providers/*`.
- **Đổi cách benchmark/tuning** → `backtest/runner.py`, `backtest/tuner.py`, `backtest/engine.py`.
- **Đổi dashboard/report** → `viz/*`, `backtest/report.py`.

## 6. Quy trình dev chuẩn cho mọi thay đổi

1. xác định scope module cần sửa,
2. thêm/sửa test trước hoặc song song,
3. chạy test nhanh:
   - `conda run -n synth python -m pytest refactor_tests -q`
4. chạy backtest scan tối thiểu:
   - `PYTHONPATH=. conda run -n synth python -m synth.miner.run_strategy_scan --assets BTC --frequencies high low --num-runs 3 --num-sims 50 --result-dir result/dev_scan`
5. nếu thay đổi routing:
   - export + validate config.
6. chạy full test:
   - `conda run -n synth python -m pytest -q`

## 7. Regression checklist trước merge

- Runtime:
  - miner vẫn gọi `entry.py`.
  - fallback chain vẫn chạy khi strategy chính fail.
- Data:
  - không leak dữ liệu tương lai.
  - provider mới không làm rỗng historical feed.
- Strategy:
  - output shape đúng.
  - không NaN/Inf.
- Backtest:
  - scan chạy hết assets/frequencies mục tiêu.
  - artifact JSON được export.
- Deploy:
  - `strategies.yaml` validate pass.
  - strategy names tồn tại trong registry.
- Viz:
  - chart/report vẫn generate.

## 8. Các lỗi thường gặp và cách xử lý nhanh

- **No data for asset/timeframe**
  - kiểm tra `data_handler.load_price_data`,
  - chạy backfill/fetch daemon.
- **Strategy not found**
  - kiểm tra `name` trong class strategy,
  - kiểm tra auto-discovery registry.
- **YAML valid nhưng route không đổi**
  - kiểm tra mtime/reload path trong `strategy_store.py`,
  - xác nhận đang dùng đúng file `config/strategies.yaml`.
- **Backtest score tốt nhưng runtime fail**
  - thường do format predictions không pass validate,
  - kiểm tra conversion path trong `entry.py` + `my_simulation.py`.

## 9. Khuyến nghị cho developer mới

- Tuần đầu chỉ nên chỉnh ở:
  - `strategies/*`,
  - `backtest/*`,
  - `config/strategies.yaml` qua deploy flow.
- Tránh sửa sớm:
  - `entry.py`,
  - `my_simulation.py`,
  - `fetch_daemon.py`,
  trừ khi đã hiểu rõ impact runtime.
- Mọi thay đổi strategy đều cần benchmark trên BTC high + low trước khi mở rộng asset khác.
