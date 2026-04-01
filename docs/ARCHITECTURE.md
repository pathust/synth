# Cấu Trúc Hệ Thống (Architecture)

## System Overview
Hệ thống Synth Miner là một nền tảng trading algorithmic phi tập trung trên mạng lưới Bittensor (Subnet 50), chuyên cung cấp các dự báo chuỗi thời gian tài chính (financial time-series) cho đa dạng tài sản (Crypto, Equity, Forex). Hệ thống được thiết kế với kiến trúc modular, tách biệt hoàn toàn giữa các tiến trình (process isolation) như thu thập dữ liệu (fetcher), huấn luyện mô hình (tuner/trainer), mô phỏng backtest và xử lý live request (miner). Sự tách biệt này giúp tối ưu hóa khả năng mở rộng, tránh xung đột tài nguyên (đặc biệt là I/O trên DuckDB), đồng thời đảm bảo miner luôn phản hồi với độ trễ thấp nhất nhờ cơ chế fallback (degradation chain) và zero-downtime hot-reload.

## Component Map
Biểu đồ luồng dữ liệu chính xác thể hiện quá trình tương tác giữa các component trong hệ thống, tập trung vào luồng xử lý Request/Response.

```mermaid
graph TD
    subgraph Bittensor Network
        VAL[Validator]
    end

    subgraph Miner Process
        AX[Axon Server]
        ROUTER[Request Router]
        
        subgraph Pipeline
            LFT[LFT Pipeline - 24h]
            HFT[HFT Pipeline - 1h]
        end
        
        REGIME[Regime Detector]
        STRAT[Strategy Selector]
        MC[Monte Carlo Engine]
        FMT[Formatter & Output Validator]
    end

    VAL -->|Synapse Request| AX
    AX -->|Parse & Authenticate| ROUTER
    ROUTER -->|Route by Horizon| LFT
    ROUTER -->|Route by Horizon| HFT
    
    LFT --> REGIME
    HFT --> REGIME
    
    REGIME --> STRAT
    STRAT --> MC
    MC --> FMT
    FMT -->|Validated Paths| AX
    AX -->|Synapse Response| VAL
```

## Data Flow — Request Lifecycle
Luồng xử lý từ khi Validator gửi yêu cầu đến khi Miner trả về kết quả:
1. **Tiếp nhận (Axon Server)**: Axon nhận Synapse request từ Validator (bao gồm thông tin tài sản, horizon, n_sim).
2. **Định tuyến (Request Router)**: Xác định loại request (LFT - 24h hoặc HFT - 1h) và phân luồng tương ứng.
3. **Phân tích bối cảnh (Regime Detector)**: Truy xuất các feature mới nhất từ bộ nhớ cache/in-memory (đã được fetcher nạp) để xác định trạng thái thị trường (bull, bear, high-vol, v.v.).
4. **Chọn chiến thuật (Strategy Selector)**: Dựa trên Asset và Regime hiện tại, tra cứu `StrategyRegistry` để lấy ra chiến thuật (hoặc ensemble) tối ưu nhất được định nghĩa trong `strategies.yaml`.
5. **Mô phỏng (Monte Carlo Engine)**: Khởi chạy các model đã chọn (Heston, GARCH, LSTM...) để sinh ra `n_sim` kịch bản giá (price paths). Quá trình này được vector hóa để tối ưu tốc độ.
6. **Định dạng & Kiểm tra (Formatter & Output Validator)**: Chuẩn hóa output (làm tròn 8 chữ số thập phân, đảm bảo không có NaN/Inf), giới hạn thời gian thực thi (timeout guard).
7. **Phản hồi (Axon Server)**: Axon trả kết quả đã format về lại cho Validator.

## Storage Architecture
Để giải quyết bài toán "DuckDB không hỗ trợ concurrent write", kiến trúc lưu trữ được phân chia nghiêm ngặt:

- **DuckDB (Market Data DB)**: Chỉ dùng để lưu trữ raw tick data và OHLCV. 
  - *Writer*: Chỉ có process `pyth_fetcher` (hoặc các data fetcher daemon) được phép ghi (write) vào DB này.
  - *Reader*: Tất cả các process khác (Miner, Tuner, Backtest) chỉ được phép kết nối với chế độ read-only.
- **Parquet (Feature Store)**: Lưu trữ các đặc trưng (features) đã được tính toán trước phục vụ training/tuner. Parquet file cho phép đọc ghi hiệu quả, có thể tách theo từng file/phân vùng (partition) nên không bị lock.
- **Per-session Storage (Backtest DB)**: Mỗi phiên backtest hoặc tuning sẽ sinh ra một file SQLite hoặc DuckDB tạm thời (ví dụ: `backtest_run_<id>.db`). Tránh hoàn toàn việc các test run đồng thời tranh chấp ghi vào một DB chung.
- **Lý do tách biệt**: DuckDB được tối ưu cho OLAP (Analytical) nhưng lại sử dụng cơ chế file lock khắt khe (1 writer, multiple readers). Việc tách biệt không gian ghi của Fetcher, Tuner và Backtest giúp hệ thống không bao giờ bị crash `database is locked` khi chạy song song.

## Process Isolation Model

| Process / Thread | Target Storage | Ghi chú (Write Permissions) |
|------------------|----------------|------------------------------|
| **Data Fetcher** (Daemon) | `market_data.duckdb` | Ghi liên tục mỗi 60s. Process duy nhất giữ Write Lock trên DuckDB chính. |
| **Live Miner** (Axon) | In-memory Cache, Logs | Chỉ Read từ DuckDB. Không ghi DB. Ghi log ra file. |
| **Tuner / Trainer** | Parquet Feature Store, `strategies.yaml` | Read từ DuckDB. Ghi feature ra Parquet. Ghi config bằng atomic swap. |
| **Backtest Engine** | `backtest_<session>.duckdb` | Tạo file DB riêng biệt trên mỗi session để ghi kết quả backtest. |

## Config Reload Lifecycle
Giải quyết vấn đề race condition khi Tuner cập nhật `strategies.yaml` trong khi Miner đang đọc.

```mermaid
sequenceDiagram
    participant T as Tuner
    participant FS as File System
    participant M as Live Miner
    
    T->>T: Tính toán tham số mới
    T->>FS: Ghi ra file tạm `strategies_tmp.yaml`
    T->>FS: Kiểm tra tính toàn vẹn file `strategies_tmp.yaml`
    T->>FS: Atomic Rename `strategies_tmp.yaml` -> `strategies.yaml`
    FS-->>M: Kích hoạt sự kiện File Modified (Watchdog)
    M->>M: Parse `strategies.yaml` vào Config_New
    M->>M: Swap reference: Config_Active = Config_New
    note over M: Zero-downtime hot reload thành công
```

## Degradation Chain
Cơ chế Fallback đảm bảo Miner không bao giờ bị timeout và luôn trả về một kết quả hợp lệ cho Validator dù model có gặp lỗi (OOM, diverge, crash).

| Level | Điều kiện kích hoạt (Trigger) | Models sử dụng | Expected Latency | Output Format |
|-------|-------------------------------|----------------|------------------|---------------|
| **L1 (Primary)** | Normal condition. | Heston, LSTM, HMM + GJR-GARCH | 100ms - 500ms | 1000 paths chuẩn |
| **L2 (Fallback 1)** | Model L1 OOM, Diverge hoặc chạy quá 600ms. | Statistical Models (GARCH-1-1, ARIMA) | < 100ms | 1000 paths chuẩn |
| **L3 (Fallback 2)** | Model L2 lỗi hoặc thiếu dữ liệu lịch sử dài. | Geometric Brownian Motion (GBM) | < 10ms | 1000 paths chuẩn |
| **L4 (Emergency)** | Tất cả L3 lỗi, CPU quá tải, sát giờ timeout. | Flatline (Giữ nguyên giá hiện tại) | < 1ms | 1000 paths không đổi |
