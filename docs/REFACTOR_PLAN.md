# Kế Hoạch Refactor (Refactor Plan)

## Folder Structure

```text
synth-miner/
├── config/             # Chứa file YAML config (system, strategies, assets)
├── data/
│   ├── fetchers/       # Tách riêng fetch logic (pyth, binance)
│   ├── storage/        # DuckDB wrapper, in-memory cache
│   └── loaders/        # OHLCV aggregation, feature engineering
├── strategies/         
│   ├── statistical/    # GARCH, ARIMA
│   ├── stochastic/     # Heston, Jump Diffusion, OU
│   ├── ml/             # LSTM, XGBoost
│   ├── regime/         # HMM, Pattern Detector
│   ├── ensemble/       # Trộn mô hình
│   ├── base.py         # BaseStrategy class
│   └── registry.py     # StrategyRegistry (Instance-based)
├── simulation/         # Engine chạy Monte Carlo và format output
├── backtest/           # Framework test chiến thuật
├── training/           # Optuna Tuner, Feature Store
├── visualization/      # Streamlit Dashboard
├── neurons/            
│   └── miner.py        # Entry point của mạng Bittensor
└── tests/              # Pytest cho toàn bộ component
```

## Migration Steps

| Bước | Tên Bước | File / Module Liên Quan | Dependency | Verify (Cách kiểm tra) |
|------|----------|-------------------------|------------|------------------------|
| 1 | **Tái cấu trúc Storage** | `data/storage/` | N/A | Chạy test `test_concurrent_duckdb` không lỗi `database is locked`. |
| 2 | **Cập nhật BaseStrategy** | `strategies/base.py` | Bước 1 | Chạy `pytest tests/test_base_strategy.py` pass. |
| 3 | **Cấu trúc lại Registry** | `strategies/registry.py` | Bước 2 | Khởi tạo 2 registry độc lập trong test và kiểm tra không ảnh hưởng chéo. |
| 4 | **Refactor Strategy Hiện Tại** | `strategies/statistical/*` | Bước 3 | So sánh output của code cũ và mới với cùng seed (độ lệch < 1e-6). |
| 5 | **Tích hợp Hot-reload** | `config/strategies.yaml`, `miner.py` | Bước 4 | Đổi file YAML khi miner đang chạy, kiểm tra log xem config mới có được load không (zero-downtime). |
| 6 | **Tích hợp Degradation Path** | `simulation/engine.py` | Bước 4 | Mock model L1 lỗi (raise Exception), kiểm tra hệ thống có tự động chạy L2 và trả về output hợp lệ hay không. |

## Breaking Changes

1. **Thay đổi cách đăng ký Strategy (Registry)**
   - *Code cũ*: Kế thừa `BaseStrategy` và dùng auto-discovery tự động nạp vào biến global `_registry`.
   - *Migration*: Khởi tạo `registry = StrategyRegistry()` tại hàm main, dùng `registry.register(MyStrategy())`.
2. **File config YAML**
   - *Code cũ*: Đọc config từ file env hoặc hardcode.
   - *Migration*: Chuyển toàn bộ cấu hình chiến thuật qua `config/strategies.yaml`. Code sẽ fail nếu file này thiếu.
3. **Database Concurrency**
   - *Code cũ*: Backtest engine và Fetcher cùng ghi vào `price_data.db`.
   - *Migration*: Fetcher ghi vào `market_data.duckdb`. Backtest engine phải dùng database riêng trong bộ nhớ hoặc tạo file tạm.

## Rollback Plan
- **Nếu refactor Storage thất bại (Bước 1)**: Rollback về commit trước đó, sử dụng SQLite như cũ.
- **Nếu refactor Strategy thất bại (Bước 4)**: Phục hồi folder `strategies/` cũ. Sử dụng adapter pattern để bọc các strategy cũ cho tương thích với engine mới tạm thời.
- **Biện pháp chung**: Toàn bộ refactor được thực hiện trên nhánh `feature/v2-architecture`. Chỉ merge vào `main` khi coverage đạt > 80% và backtest không suy giảm (CRPS không tăng).
