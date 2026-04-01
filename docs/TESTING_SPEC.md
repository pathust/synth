# Testing Specification

## Test Strategy
Chiến lược kiểm thử chia thành 3 cấp độ:
- **Unit Test**: Test độc lập các hàm toán học (CRPS calculation), format output, config parser. 
- **Integration Test**: Test luồng dữ liệu (DB -> Feature -> Model -> Output), quá trình hot-reload, và fallback chain.
- **Backtest Scenario**: Chạy mô phỏng trên dữ liệu lịch sử dài để đánh giá hiệu quả tài chính (CRPS score).

## Fixtures Cần Có
Các pytest fixtures được định nghĩa trong `conftest.py`:

- `fresh_registry`: Trả về một instance `StrategyRegistry` mới tinh, không chứa bất kỳ strategy nào.
- `mock_db`: Tạo in-memory DuckDB connection, nạp sẵn 1000 dòng dữ liệu OHLCV giả lập.
- `sample_ohlcv`: Pandas DataFrame chứa 100 nến OHLCV của BTC để test logic mô hình.
- `mock_config`: Tạo file `strategies.yaml` tạm thời trong thư mục temp và xóa đi sau khi test xong.

## Critical Test Cases

### 1. Concurrent Write Test (DuckDB Isolation)
**REQ-001**: Đảm bảo Fetcher và Backtester không bị lock database.
- **Given**: Fetcher đang có write-lock trên `market_data.duckdb`.
- **When**: Backtest Engine cố gắng khởi tạo (tạo db riêng `backtest.duckdb` và đọc từ `market_data.duckdb`).
- **Then**: Không xảy ra lỗi `database is locked`, Backtest lấy được data.

### 2. Hot-reload Race Condition Test
**REQ-002**: Đảm bảo Miner không bao giờ đọc phải file config hỏng.
- **Given**: Miner đang chạy vòng lặp mô phỏng.
- **When**: Tuner ghi file `strategies_new.yaml` và thực hiện `os.replace()` sang `strategies.yaml`.
- **Then**: Miner load thành công config mới ở chu kỳ tiếp theo mà không văng lỗi `YAML parse error` giữa chừng.

### 3. Fallback Chain Test (Mock Model Failure)
**REQ-003**: Đảm bảo Miner luôn trả về kết quả hợp lệ ngay cả khi model OOM/Crash.
- **Given**: Request từ Validator gọi model L1 (LSTM).
- **When**: Mock model L1 `raise MemoryError("OOM")`.
- **Then**: Hệ thống bắt được exception, tự động fallback xuống L2 (GARCH), và trả về mảng shape `(1000, N)` đúng thời hạn.

### 4. Registry Isolation Test (No Test Pollution)
**REQ-004**: Đảm bảo các test không chia sẻ state của StrategyRegistry.
- **Given**: Test A đăng ký `TestStrategy1` vào `fresh_registry`.
- **When**: Test B khởi chạy với `fresh_registry` mới.
- **Then**: Test B không nhìn thấy `TestStrategy1`.

## Backtest Scenario Template
Một kịch bản backtest (scenario) được lưu dưới định dạng YAML để dễ dàng chia sẻ và tái lập:

```yaml
scenario_name: "BTC_Flash_Crash_May2021"
description: "Test khả năng phản ứng của Heston model trong cú sập giá BTC tháng 5/2021"
asset: "BTC"
timeframe: "1h"
date_range:
  start: "2021-05-15T00:00:00Z"
  end: "2021-05-25T00:00:00Z"
strategies:
  - name: "heston_stochastic"
    params:
      kappa: 2.0
      theta: 0.1
      sigma: 0.5
  - name: "garch_baseline"
    params:
      p: 1
      q: 1
evaluation_metrics:
  - "mean_crps"
  - "max_drawdown"
pass_condition:
  mean_crps_less_than: 0.005
```
