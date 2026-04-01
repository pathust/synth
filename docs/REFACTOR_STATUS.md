# Refactor Status

## Phạm vi đã triển khai

- Miner runtime đã chuyển sang dùng entry thống nhất: `neurons/miner.py` gọi `synth.miner.entry.generate_simulations`.
- Cấu hình chiến lược đã được chuẩn hóa theo YAML schema tại `synth/miner/config/strategies.yaml`.
- Thêm `StrategyStore` để:
  - đọc config YAML,
  - map `routing` theo `(asset, regime)`,
  - tự reload khi file thay đổi bằng cơ chế kiểm tra `mtime`,
  - fallback an toàn về cấu hình Python cũ nếu YAML lỗi hoặc thiếu.
- Entry pipeline đã dùng trực tiếp `StrategyStore` cho:
  - chọn danh sách strategy theo asset/time_length,
  - lấy fallback chain theo config.
- Thêm lớp lưu trữ DuckDB và provider mới:
  - `synth/miner/data/storage/duckdb_store.py` (writer/read-only reader),
  - `synth/miner/data/storage/duckdb_sync.py` (đồng bộ dữ liệu từ dict giá sang DuckDB),
  - `synth/miner/pipeline/providers/duckdb_provider.py`,
  - `PriceLoader` ưu tiên DuckDB trước MySQL.
- `fetch_daemon.py` đã thêm bước mirror dữ liệu giá sang DuckDB sau mỗi vòng fetch (có thể bật/tắt bằng `SYNTH_ENABLE_DUCKDB_WRITER`).
- Hợp nhất API backtest:
  - `synth/miner/backtest/engine.py` bổ sung `run_slots(...)` cho luồng slot-based,
  - `synth/miner/backtest/framework.py` chuyển thành lớp tương thích (delegate sang engine thống nhất),
  - các script backtest chính đã chuyển import sang `synth.miner.backtest.engine.BacktestEngine`.
- Chuẩn hóa package export `synth.miner.backtest` bằng lazy import cho `BacktestEngine` và `ExperimentConfig`.
- Bổ sung hàm metric nền tảng còn thiếu (`METRICS`, `compute_crps_score`, `compute_var`, `compute_es`) trong `synth/miner/backtest/metrics.py` để runner/duel hoạt động nhất quán.
- Chuẩn hóa deploy config sang YAML runtime:
  - `synth/miner/deploy/exporter.py` xuất trực tiếp `synth/miner/config/strategies.yaml`,
  - ghi file theo atomic swap (`.tmp` + `os.replace`) để tương thích hot-reload an toàn,
  - `synth/miner/deploy/applier.py` validate schema YAML + strategy name theo registry.
- Ổn định compatibility entrypoint:
  - `synth/miner/run.py` dùng `synth.miner.entry.generate_simulations`,
  - `synth/miner/entry_new.py` chuyển thành wrapper deprecate và delegate sang entry chuẩn.

## Kiểm thử đã bổ sung

- `refactor_tests/test_strategy_store.py`
  - kiểm tra parse YAML đúng schema,
  - kiểm tra reload sau `os.replace` (atomic swap pattern).
- `refactor_tests/test_registry_isolation.py`
  - kiểm tra 2 instance `StrategyRegistry` không chia sẻ state.
- `refactor_tests/test_duckdb_store.py`
  - kiểm tra write/read batch cho `raw_prices`,
  - kiểm tra `DuckDBProvider` fetch được dữ liệu theo khoảng thời gian.
- `refactor_tests/test_duckdb_sync.py`
  - kiểm tra đồng bộ dict giá vào DuckDB với giới hạn số dòng.
- `refactor_tests/test_backtest_unification.py`
  - kiểm tra `engine.run_slots(...)` hoạt động,
  - kiểm tra `framework.BacktestEngine` delegate đúng sang engine mới.
- `refactor_tests/test_deploy_yaml.py`
  - kiểm tra exporter ghi đúng schema `strategies.yaml`,
  - kiểm tra applier validate được config YAML hợp lệ.

## Cách chạy test trong môi trường yêu cầu

```bash
conda run -n synth python -m pytest refactor_tests -q
```

## Trạng thái test hiện tại

- `refactor_tests`: pass.
- Full suite `pytest -q`: chạy thành công trong môi trường hiện tại (`70 passed, 18 skipped, 2 subtests passed`).
- Chính sách fixture DB đã được chuyển sang on-demand:
  - `tests/conftest.py` không còn `autouse` cho DB setup/migration.
  - Chỉ các test thật sự dùng `db_engine` mới yêu cầu Docker/Testcontainers.
- Tương thích Python runtime mới:
  - `synth/base/dendrite.py` và `synth/base/dendrite_multiprocess.py` đã fallback an toàn khi `uvloop` không tương thích.
  - thêm `synth/miner/simulations.py` làm compatibility module cho các test/import cũ.
- Cảnh báo runtime đã được giảm mạnh:
  - thay `asyncio.iscoroutinefunction` bằng `inspect.iscoroutinefunction`,
  - đổi `SimulationInput` sang `ConfigDict` của Pydantic v2,
  - loại bỏ các `return` trong `finally` gây `SyntaxWarning`.
  - hiện full test suite còn 1 warning deprecation có chủ đích từ lớp compatibility `backtest.framework.BacktestEngine`.
