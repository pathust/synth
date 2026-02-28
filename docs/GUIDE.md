# Synth Miner — Tài Liệu Hướng Dẫn Chi Tiết

> Tài liệu này bao gồm: cài đặt, nguyên lý hoạt động, cách update/sửa/thay thế,
> và hướng dẫn chi tiết cho phần simulation và backtest.

---

## Mục Lục

1. [Cài Đặt & Thiết Lập](#1-cài-đặt--thiết-lập)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Nguyên Lý Hoạt Động — Simulation](#3-nguyên-lý-hoạt-động--simulation)
4. [Nguyên Lý Hoạt Động — Data Pipeline](#4-nguyên-lý-hoạt-động--data-pipeline)
5. [Hướng Dẫn Backtest](#5-hướng-dẫn-backtest)
6. [Cách Update / Sửa / Thay Thế](#6-cách-update--sửa--thay-thế)
7. [Cấu Hình & Hyperparameters](#7-cấu-hình--hyperparameters)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Cài Đặt & Thiết Lập

### 1.1 Yêu cầu hệ thống

- Python 3.10+ (đã test trên 3.14)
- conda (Miniconda hoặc Anaconda)
- Khoảng 2GB disk cho data + packages

### 1.2 Cài đặt môi trường

```bash
# Tạo conda environment
conda create -n synth python=3.14 -y
conda activate synth

# Cài đặt dependencies
cd /path/to/synth
pip install -r requirements.txt
```

### 1.3 Các packages quan trọng

| Package | Mục đích |
|---------|----------|
| `arch` | Fitting GARCH/APARCH models |
| `statsmodels` | HAR-RV regression |
| `scipy` | Student-t distribution sampling |
| `coinmetrics-api-client` | CoinMetrics API (historical prices) |
| `matplotlib` | Visualization, charts |
| `numpy`, `pandas` | Xử lý dữ liệu số |

### 1.4 Cấu trúc dữ liệu

Hệ thống sử dụng **SQLite** để lưu price data locally (không cần Docker/MySQL cho backtest):

```
synth/miner/data/price_data.db    ← Tự động tạo khi chạy lần đầu
```

Nếu muốn reset data: xóa file `price_data.db`, hệ thống sẽ tự fetch lại từ API.

### 1.5 Biến môi trường (tùy chọn)

Xem `.env.example` cho danh sách đầy đủ. Cho backtest local, **không cần cấu hình gì thêm** — tất cả đều có giá trị mặc định.

---

## 2. Kiến Trúc Hệ Thống

### 2.1 Sơ đồ tổng quan

```
                        ┌─────────────────────────┐
                        │     backtest_runner.py   │  ← Entry point
                        │  backtest_enhanced.py    │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │     simulations.py       │  ← Điều phối model
                        │  generate_simulations()  │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼───────────────────────┐
              │                      │                       │
    ┌─────────▼─────────┐  ┌────────▼────────┐  ┌──────────▼──────────┐
    │  my_simulation.py  │  │  core/ models   │  │ price_simulation.py │
    │  fetch_price_data  │  │  GARCH, HAR-RV  │  │  get_asset_price    │
    │  simulate_paths    │  │  APARCH, etc.   │  │  (live Pyth price)  │
    └─────────┬─────────┘  └─────────────────┘  └─────────────────────┘
              │
    ┌─────────▼─────────┐
    │  data_handler.py   │  ← Fetch + Store
    │  DataHandler       │
    └──────┬──────┬──────┘
           │      │
    ┌──────▼──┐ ┌─▼──────────────┐
    │ SQLite  │ │ CoinMetrics    │
    │ (local) │ │ + Pyth API     │
    └─────────┘ └────────────────┘
```

### 2.2 Cấu trúc file

```
synth/miner/
├── simulations.py          # Điều phối: chọn model → chạy simulation
├── my_simulation.py        # Fetch data + dispatch simulation function
├── data_handler.py         # Fetch từ API (CM + Pyth), lưu SQLite
├── mysql_handler.py        # SQLite handler (save/load price data)
├── coinmetric_client.py    # CoinMetrics API client
├── constants.py            # Danh sách assets
├── compute_score.py        # CRPS scoring (so sánh prediction vs thực tế)
├── run.py                  # Benchmark test runner
├── backtest_runner.py      # Basic backtest (multi-asset)
├── backtest_enhanced.py    # Enhanced backtest (multi-model + visualization)
├── plot_chart.py           # Chart utilities
│
├── core/                   # Các thuật toán simulation
│   ├── garch_simulator.py      # GARCH v1 (basic, Constant mean)
│   ├── grach_simulator_v2.py   # GARCH v2 (optimized, Zero mean) ★ Recommended
│   ├── grach_simulator_v2_1.py # GARCH v2.1 (XAU specialized)
│   ├── HAR_RV_simulatior.py    # HAR Realized Variance + GARCH
│   ├── aparch_simulator.py     # APARCH + regime detection
│   ├── heston_jump_simulator.py # Heston + Jump Diffusion
│   ├── spyx_simulator.py       # SPYX (S&P 500 ETF)
│   ├── stock_simulator.py      # Stock with intraday seasonality
│   ├── stock_simulator_v2.py   # Weekly seasonal optimized
│   └── regime_detection.py     # Market regime detection (BBW, ER)
│
└── data/
    └── price_data.db       # SQLite database (auto-created)
```

---

## 3. Nguyên Lý Hoạt Động — Simulation

### 3.1 Pipeline chính

Khi gọi `generate_simulations()`, hệ thống thực hiện:

```
1. Nhận SimulationInput (asset, start_time, time_increment, time_length, num_simulations)
2. Theo asset → chọn danh sách model ưu tiên (fallback chain)
3. Với mỗi model trong chain:
   a. Gọi simulate_crypto_price_paths() → fetch data + chạy model
   b. Validate format → nếu OK thì trả kết quả
   c. Nếu lỗi → thử model tiếp theo
4. Trả về predictions = [timestamps, base_prices, path_1, path_2, ...]
```

### 3.2 Fallback chain theo asset

| Asset | Model 1 (ưu tiên) | Model 2 | Model 3 |
|-------|-------------------|---------|---------|
| BTC, ETH, SOL | GARCH v2 | GARCH v1 | HAR-RV |
| XAU (v2_1_0) | GARCH v2.1 | — | — |
| NVDAX | Seasonal Stock | — | — |
| TSLAX, AAPLX, GOOGLX, XAU | Weekly Seasonal | — | — |

**Để thay đổi**: sửa `simulations.py` → hàm `generate_simulations()` → biến `lst_simulate_fn`.

### 3.3 Thuật toán GARCH(1,1)

**Đây là model chính**, hoạt động như sau:

#### Bước 1: Chuẩn bị dữ liệu
```python
# Lấy giá lịch sử từ SQLite/API
prices = [65000, 65100, 64900, ...]  # Chuỗi giá 5-phút

# Tính log-returns
log_returns = log(P_t / P_{t-1})  # Chuỗi lợi suất
```

#### Bước 2: Fit GARCH model
```python
# Model: r_t = μ + ε_t, ε_t = σ_t · z_t
# σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
# z_t ~ Student-t(ν)

# Tham số cần fit: μ, ω, α, β, ν
# α: phản ứng với shock (cao = nhạy hơn với biến động đột ngột)
# β: persistence (cao = volatility duy trì lâu)
# ν: degrees of freedom (thấp = đuôi dày, nhiều extreme events)
```

#### Bước 3: Simulate paths
```python
# Với mỗi step t = 1..steps:
#   1. Tính σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
#   2. Sample z_t từ Student-t(ν)
#   3. ε_t = σ_t · z_t
#   4. r_t = μ + ε_t
#   5. P_t = P_{t-1} · exp(r_t)
```

#### Sự khác nhau giữa các phiên bản GARCH

| Feature | GARCH v1 | GARCH v2 | HAR-RV | APARCH |
|---------|----------|----------|--------|--------|
| Mean model | Constant | **Zero** | Constant | Constant |
| Lookback | Toàn bộ data | **45 ngày** (configurable) | 30+ ngày | 45 ngày + regime |
| Volatility init | Unconditional | **Last conditional** | HAR forecast | Last + regime adjust |
| Scale | 10000 (bps) | 10000 | 10000 | 10000 |
| Regime-aware | ❌ | ❌ | ❌ | ✅ (BBW squeeze/trend) |
| Asymmetry | ❌ | ❌ | ❌ | ✅ (γ parameter) |
| Best for | Simple case | **Crypto chung** | Cần >30 ngày data | Volatile markets |

### 3.4 APARCH — Model nâng cao

APARCH thêm 2 tham số so với GARCH:
- **γ (gamma)**: Leverage effect — phản ứng khác nhau với shock dương vs âm
- **δ (delta)**: Power parameter — thay đổi dạng hàm phi tuyến

```
σ^δ_t = ω + α·(|ε_{t-1}| - γ·ε_{t-1})^δ + β·σ^δ_{t-1}
```

Khi `δ=2` và `γ=0`, APARCH giản hóa thành GARCH thường.

APARCH kết hợp với **regime detection**:
- **BBW Squeeze**: Bollinger Band Width thấp → thị trường đi ngang → giảm vol_multiplier
- **Trending**: Efficiency Ratio cao → thị trường có xu hướng → tăng vol_multiplier

---

## 4. Nguyên Lý Hoạt Động — Data Pipeline

### 4.1 Luồng fetch data

```
fetch_price_data(asset, time_increment)
│
├── Kiểm tra SQLite: có data cho asset/timeframe không?
│   ├── CÓ → kiểm tra staleness (data cũ hơn 1 interval?)
│   │   ├── CÓ → fetch thêm data mới từ API
│   │   └── KHÔNG → trả data hiện có
│   └── KHÔNG → fetch toàn bộ 14 ngày lịch sử
│
└── fetch_multi_timeframes_price_data()
    ├── Chia thành batches (1000 points/batch)
    ├── Với mỗi batch, race 2 nguồn:
    │   ├── CoinMetrics API (historical)
    │   └── Pyth API via PriceDataProvider
    ├── Lấy kết quả nhanh hơn, hủy cái kia
    └── Lưu vào SQLite
```

### 4.2 Data sources

| Source | Endpoint | Ưu điểm | Hạn chế |
|--------|----------|---------|---------|
| CoinMetrics | `api.coinmetrics.io/v4` | Dữ liệu lịch sử sâu | Free tier giới hạn |
| Pyth Network | `benchmarks.pyth.network` | Real-time, decentralized | Lịch sử ~14 ngày |

### 4.3 Cách data được lưu trong SQLite

```sql
-- Bảng price_data
CREATE TABLE price_data (
    id INTEGER PRIMARY KEY,
    asset TEXT,
    time_frame TEXT,
    data TEXT  -- JSON: {"timestamp1": "price1", "timestamp2": "price2", ...}
)
-- Index: (asset, time_frame)
```

**Truy vấn thủ công**:
```bash
sqlite3 synth/miner/data/price_data.db "SELECT asset, time_frame, length(data) FROM price_data;"
```

---

## 5. Hướng Dẫn Backtest

### 5.1 Chạy backtest cơ bản (đánh giá nhanh)

```bash
cd /path/to/synth
conda activate synth

# Chạy 15 tests (3 assets × 5 dates)
python -m synth.miner.backtest_runner
```

**Output**: `result/` chứa JSON results + log.

### 5.2 Chạy backtest nâng cao (so sánh models)

```bash
# Chạy 60 tests (4 models × 3 assets × 5 dates) + visualize
python -m synth.miner.backtest_enhanced
```

**Output**: `result/` chứa JSON results + `result/charts/` chứa PNG charts + log.

### 5.3 Tùy chỉnh backtest

Mở file `synth/miner/backtest_runner.py` hoặc `backtest_enhanced.py`, sửa phần Config:

```python
# ── Config ───────────────────────────────────────────
ASSETS = ["BTC", "ETH", "SOL"]    # Thêm/bớt assets
NUM_TEST_DATES = 5                 # Số ngày test
DAYS_BACK_START = 10               # Bắt đầu từ N ngày trước
DAYS_BACK_END = 3                  # Kết thúc trước N ngày
TIME_INCREMENT = 300               # 300s = 5 phút
TIME_LENGTH = 3600                 # 3600s = 1 giờ simulation
NUM_SIMULATIONS = 100              # Số paths mô phỏng
SEED = 42                         # Random seed (đổi để test khác)
```

#### Ví dụ: Backtest 24 giờ với BTC

```python
ASSETS = ["BTC"]
TIME_LENGTH = 86400        # 24 giờ
NUM_SIMULATIONS = 1000     # Nhiều paths hơn cho chính xác
NUM_TEST_DATES = 10        # 10 ngày test
```

#### Ví dụ: Test model cụ thể

Trong `backtest_enhanced.py`:
```python
# Chỉ test 2 models
MODELS = {
    "GARCH_v2": garch_v2,
    "APARCH":   aparch,
}
```

### 5.4 Đọc kết quả

```json
// result/BTC_results.json
[
  {
    "asset": "BTC",
    "start_time": "2026-02-18T21:13:00+00:00",
    "time_increment": 300,
    "elapsed_seconds": 0.77,     // Thời gian chạy
    "num_paths": 100,            // Số đường simulation
    "path_length": 13,           // Số điểm trên mỗi path
    "price_start": 66158.91,     // Giá bắt đầu
    "price_end": 66269.02,       // Giá trung bình cuối (mean)
    "format_valid": true,        // Prediction hợp lệ
    "status": "SUCCESS",
    "predictions_sample": [...]  // Mẫu 3 paths đầu, 5 điểm đầu
  }
]
```

### 5.5 Đánh giá mô hình

Hai chỉ số chính:

1. **σ_end (Std Dev cuối)**: Đo độ phân tán predictions
   - **Thấp hơn = tốt hơn** cho CRPS scoring
   - GARCH_v2: σ ≈ 160, APARCH: σ ≈ 164, HAR_RV: σ ≈ 492

2. **CRPS Score**: Continuous Ranked Probability Score
   - So sánh phân phối prediction với giá thực tế
   - **Thấp hơn = chính xác hơn**
   - Cần real prices (từ Pyth API) để tính → dùng `run.py` benchmark_test

---

## 6. Cách Update / Sửa / Thay Thế

### 6.1 Thêm model simulation mới

**Bước 1**: Tạo file mới trong `synth/miner/core/`

```python
# synth/miner/core/my_new_model.py

import numpy as np
import pandas as pd
from typing import Optional

def simulate_my_new_model(
    prices_dict: dict,      # {"timestamp": "price", ...}
    asset: str,             # "BTC", "ETH", ...
    time_increment: int,    # 300 (5m) hoặc 60 (1m)
    time_length: int,       # 3600 (1h) hoặc 86400 (24h)
    n_sims: int,            # Số paths
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Returns: ndarray shape (n_sims, steps+1)
    - Cột 0: giá hiện tại (S0)
    - Cột 1..steps: giá mô phỏng
    """
    # 1. Parse dữ liệu
    timestamps = sorted(prices_dict.keys())
    prices = pd.Series(
        [float(prices_dict[ts]) for ts in timestamps],
        index=pd.to_datetime([int(ts) for ts in timestamps], unit='s')
    )
    S0 = float(prices.iloc[-1])
    steps = time_length // time_increment

    # 2. Logic mô phỏng của bạn ở đây
    np.random.seed(seed)
    # ... fitting model, generating paths ...
    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = S0
    # paths[:, 1:] = ...

    return paths
```

**Bước 2**: Import vào `simulations.py`

```python
# Thêm import
from synth.miner.core.my_new_model import simulate_my_new_model

# Trong generate_simulations(), thêm vào fallback chain:
if asset in ["BTC", "ETH", "SOL", "XAU"]:
    lst_simulate_fn = [
        simulate_my_new_model,           # ← Thêm ở đầu = ưu tiên cao nhất
        simulate_single_price_path_with_garch_v2,
        simulate_single_price_path_with_garch,
    ]
```

**Bước 3**: Test model mới

```python
# Thêm vào backtest_enhanced.py
MODELS = {
    "MY_MODEL": simulate_my_new_model,
    "GARCH_v2": garch_v2,
    "APARCH": aparch,
}
```

```bash
python -m synth.miner.backtest_enhanced
```

### 6.2 Thay đổi data source

File `data_handler.py` → class `DataHandler`:

```python
# Thêm nguồn dữ liệu mới
def fetch_my_source_data(self, asset, start_time, end_time, time_increment):
    """Fetch từ nguồn mới."""
    # Gọi API, parse response
    # Trả về dict: {"timestamp_unix": "price_float", ...}
    return transformed_data
```

Sau đó trong `fetch_multi_timeframes_price_data()`, thêm future mới:

```python
future_my_source = executor.submit(
    self.fetch_my_source_data, asset, start_time_i, end_time_i, inc
)
futures_list.append(future_my_source)
futures_dict[future_my_source] = "MySource"
```

### 6.3 Đổi database (SQLite → PostgreSQL/MySQL)

Sửa file `mysql_handler.py`:

```python
# Thay thế sqlite3 bằng connector khác:
import psycopg2  # cho PostgreSQL
# import pymysql  # cho MySQL

class MySQLHandler:
    def __init__(self):
        # Thay self.conn = sqlite3.connect(...)
        self.conn = psycopg2.connect(
            host="localhost", port=5432,
            dbname="synth", user="user", password="pass"
        )
```

> ⚠️ Giữ nguyên interface: `save_price_data(asset, time_frame, data)` và `load_price_data(asset, time_frame)`.

### 6.4 Thay đổi hyperparameters

File `synth/miner/core/grach_simulator_v2.py` → hàm `get_optimal_config()`:

```python
def get_optimal_config(asset: str, time_increment: int) -> dict:
    config = {
        "mean_model": "Zero",       # Sửa: "Zero" hoặc "Constant"
        "dist": "StudentsT",        # Sửa: "StudentsT" hoặc "Normal"
        "scale": 10000.0,           # Scale (giữ 10000 cho bps)
        "min_nu": 0.0,              # Min degrees of freedom
        "vol_multiplier": 1.0,      # Nhân σ (0.95 = tighter, 1.05 = wider)
        "lookback_days": 45,        # Số ngày lịch sử để fit
    }
```

### 6.5 Mở rộng cửa sổ dữ liệu lịch sử

Mặc định fetch **14 ngày** (do API limit). Để tăng:

File `my_simulation.py`:
```python
# Dòng 67-73: Thay timedelta(days=14) → timedelta(days=30)
start_time_crawl = (
    datetime.datetime.now(datetime.timezone.utc)
    .replace(second=0, microsecond=0)
    - datetime.timedelta(days=30)  # ← Đổi ở đây
)
hist_price_data = data_handler.fetch_multi_timeframes_price_data(
    asset, start_time_crawl, weeks=4, time_frame=time_frame  # ← weeks tương ứng
)
```

> ⚠️ Fetch >14 ngày 1m data từ Pyth API có thể trả về rỗng. Nên dùng 5m resolution với data dài hơn.

### 6.6 Thêm asset mới

1. Thêm vào `constants.py`:
   ```python
   ASSETS_PERIODIC_FETCH_PRICE_DATA = ["SPYX", "NVDAX", "MY_NEW_ASSET"]
   ```

2. Thêm vào routing trong `simulations.py`:
   ```python
   if asset == "MY_NEW_ASSET":
       lst_simulate_fn = [simulate_my_new_model]
       max_data_points = [None]
   ```

3. Thêm vào backtest config:
   ```python
   ASSETS = ["BTC", "ETH", "SOL", "MY_NEW_ASSET"]
   ```

---

## 7. Cấu Hình & Hyperparameters

### 7.1 Prompt configs (từ validator)

File `synth/validator/prompt_config.py`:

| Config | time_length | time_increment | num_simulations | Mô tả |
|--------|-------------|----------------|-----------------|--------|
| `LOW_FREQUENCY` | 86400 (24h) | 300 (5m) | 1000 | Task chính |
| `HIGH_FREQUENCY` | 3600 (1h) | 60 (1m) | 1000 | Task phụ |

### 7.2 GARCH v2 config per asset

| Asset | lookback_days (5m) | lookback_days (1m) | mean_model | vol_multiplier |
|-------|-------------------|-------------------|------------|----------------|
| BTC | 45 | 7 | Zero | 1.0 |
| ETH | 45 | 7 | Zero | 1.0 |
| SOL | 45 | 7 | Zero | 1.0 |
| XAU | 90 | 14 | Zero | 1.0 |
| SPYX | 45 | 7 | Zero | 0.9 |

### 7.3 APARCH config per regime

| Regime | vol_multiplier | mean_model | min_nu | Impact |
|--------|---------------|------------|--------|--------|
| Squeeze (BBW thấp) | 0.95 | Zero | 6.0 | Tighter predictions |
| Trending (ER cao) | 1.05 | Constant | 4.5 | Wider, follow trend |
| Bình thường | 0.96 | Constant | 4.0 | Balanced |

---

## 8. Troubleshooting

### 8.1 "No data returned" từ API

**Nguyên nhân**: API không có historical data cho khoảng thời gian yêu cầu.

**Giải pháp**: Giảm cửa sổ fetch hoặc đổi resolution:
```python
# my_simulation.py: giảm days
- datetime.timedelta(days=14)
+ datetime.timedelta(days=7)
```

### 8.2 "Dữ liệu lịch sử ngắn hơn mức tối ưu"

Đây là **warning**, không phải lỗi. Model vẫn chạy nhưng kém chính xác. Để fix: tăng cửa sổ data (xem 6.5).

### 8.3 GARCH overflow (nan prices)

**Nguyên nhân**: GARCH v1 với `Constant` mean trên data ngắn gây overflow trong `exp()`.

**Giải pháp**: Dùng GARCH v2 (Zero mean, numerically stable hơn). Đây đã là mặc định.

### 8.4 HAR_RV warning "quá ngắn để ước tính HAR ổn định"

HAR-RV cần ≥30 ngày data hàng ngày (Realized Variance). Với 14 ngày, nó fallback sang RV trung bình → accuracy thấp.

**Giải pháp**: Tăng data window lên 30+ ngày, hoặc không dùng HAR_RV cho short windows.

### 8.5 Reset toàn bộ data

```bash
rm -f synth/miner/data/price_data.db
rm -rf result/
```

Chạy lại backtest, hệ thống sẽ tự fetch mới.
