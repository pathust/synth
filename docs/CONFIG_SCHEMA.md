# Configuration & Schema Specification

## 1. Mục tiêu
Định nghĩa cấu trúc chuẩn cho các file cấu hình (YAML) để AI Agent và hệ thống parse không bị lỗi. Cấu trúc rõ ràng giúp cơ chế hot-reload hoạt động trơn tru.

## 2. File: `strategies.yaml`
**Mục đích**: Chứa mapping giữa (Tài sản + Trạng thái thị trường) -> (Danh sách model + Trọng số + Tham số).
**REQ-007**: File này là nguồn chân lý (Source of Truth) cho Live Miner. Tuner sẽ liên tục ghi đè file này sau mỗi lần tối ưu.

### Cấu trúc Schema
```yaml
version: "2.0"
updated_at: "2026-03-31T10:00:00Z"

# Cấu hình fallback chain (Degradation Path)
fallback_chain:
  L1_timeout_ms: 600
  L2_model: "garch_baseline"
  L3_model: "gbm_baseline"

# Mapping chiến thuật
routing:
  BTC:
    bull:
      ensemble_method: "weighted_average"
      models:
        - name: "heston_stochastic"
          weight: 0.7
          params:
            kappa: 2.0
            theta: 0.1
            sigma: 0.5
        - name: "gjr_garch"
          weight: 0.3
          params:
            p: 1
            q: 1
    bear:
      # ... tương tự cho các regime khác
  XAU:
    trending:
      # ...
```

## 3. File: `system.yaml`
**Mục đích**: Cấu hình các tham số vận hành hệ thống, port, đường dẫn file. Không hot-reload (cần restart process nếu đổi).

### Cấu trúc Schema
```yaml
environment: "production" # "production" hoặc "testing"

storage:
  duckdb_path: "data/market_data.duckdb"
  feature_store_path: "data/features/"
  backtest_temp_dir: "data/backtest_runs/"

network:
  axon_port: 8091
  validator_timeout_seconds: 12

tuning:
  optuna_n_trials: 100
  cron_schedule: "0 0 * * *" # Chạy tuner mỗi nửa đêm
```

## 4. Quá trình Hot-Reload (Atomic Swap Pattern)
Để đảm bảo Miner không đọc phải file YAML đang ghi dở (corrupt):
1. Tuner tạo ra dict Python mới.
2. Ghi dict này ra file `config/strategies.tmp.yaml`.
3. Gọi hàm kiểm tra: `yaml.safe_load('config/strategies.tmp.yaml')` để xác nhận file hợp lệ.
4. Gọi `os.replace('config/strategies.tmp.yaml', 'config/strategies.yaml')` (Đây là thao tác atomic ở mức Hệ điều hành).
5. Miner dùng `watchdog` library để lắng nghe event `FileModified` trên file `strategies.yaml` và reload vào RAM.
