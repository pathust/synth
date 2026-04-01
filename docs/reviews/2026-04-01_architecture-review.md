# Đánh Giá Kiến Trúc Code Hiện Tại

Ngày đánh giá: 2026-04-01  
Người đánh giá: Codex (xác minh mức code + lệnh runtime)

## 1. Phạm vi và cách đánh giá

- Phạm vi: `synth/miner` và các tài liệu liên quan (`docs/*`) với trọng tâm vào luồng runtime hiện tại.
- Cách đánh giá:
  - Đọc code thực thi (entrypoint, luồng dữ liệu, strategy/config/backtest orchestration).
  - Đối chiếu tài liệu kiến trúc hiện có với quá trình triển khai (implementation) thực tế.
  - Chạy một số lệnh xác thực runtime trong môi trường `conda -n synth`.

## 2. Tổng kết nhanh

Kiến trúc hiện tại đã có định hướng module hóa (entry, strategy registry, backtest runner, deploy exporter), nhưng đang gặp 3 vấn đề lớn:

1. Tài liệu kiến trúc và triển khai thực tế đang lệch nhau rõ ràng (đặc biệt là tầng dữ liệu/lưu trữ).
2. Quyết định về frequency/data resolution đang gắn theo `asset` thay vì ngữ cảnh prompt/request context.
3. Tài liệu + template cho strategy đã cũ, gây nguy cơ tạo strategy/tooling không tương thích.

## 3. Các phát hiện chi tiết (ưu tiên theo mức độ)

### [P1] Kiến trúc dữ liệu thực tế khác mô tả trong tài liệu

**Bằng chứng**

- Tài liệu mô tả DuckDB là cơ sở lưu trữ trung tâm, miner/tuner/backtest đọc từ DuckDB, backtest ghi session vào DB riêng:
  - `docs/ARCHITECTURE.md:58-63`
  - `docs/ARCHITECTURE.md:69-72`
- Code runtime thực tế đang tải/lưu giá qua MySQL (`DataHandler -> MySQLHandler`):
  - `synth/miner/data_handler.py:437-464`
  - `synth/miner/data_handler.py:466-502`
- Fetch daemon cũng ghi vào MySQL trước, DuckDB chỉ là bản sao (mirror) tùy chọn:
  - `synth/miner/fetch_daemon.py:111-128`
  - `synth/miner/fetch_daemon.py:134-148`

**Tác động**

- Đội ngũ dễ hiểu nhầm về nguồn dữ liệu "chính thống" (source of truth).
- Dễ làm sai quy trình debug (nhìn vào DuckDB trong khi runtime lại đọc từ MySQL).
- Nguy cơ lỗi vận hành khi bàn giao/onboarding.

**Khuyến nghị**

- Chốt rõ nguồn dữ liệu chính (MySQL hay DuckDB) trong 1 tài liệu RFC ngắn.
- Cập nhật `docs/ARCHITECTURE.md` theo thực tế triển khai hiện tại (hoặc cấu trúc lại - refactor code theo tài liệu nếu tài liệu là mục tiêu).
- Nếu giữ lưu trữ song song (dual-store), cần ghi rõ quyền sở hữu (ownership): bên ghi chính, bên đọc chính, và thời gian cam kết đồng bộ (SLA sync).

---

### [P1] Chọn độ phân giải thời gian (time resolution) theo tài sản, không theo ngữ cảnh request

**Bằng chứng**

- `UnifiedDataLoader` ưu tiên `HIGH_FREQUENCY` nếu tài sản có nhãn high:
  - `synth/miner/data/dataloader.py:22-26`
- `simulate_crypto_price_paths` gọi dataloader mà không truyền `time_increment/time_length`:
  - `synth/miner/my_simulation.py:141-162`

**Tác động**

- Các tài sản có cả nhãn high/low (`BTC/ETH/SOL/XAU`) có thể bị nạp dữ liệu lịch sử từng 1 phút ngay cả khi prompt là phân giải low (5 phút).
- Làm lệch calibration strategy và tạo ra độ võng giữa kết quả chạy backtest và kết quả ở môi trường live.

**Khuyến nghị**

- Dataloader API cần nhận giá trị `time_increment` (hoặc `prompt_label`) rõ ràng từ bên gọi hàm.
- Cấm mặc định lấy theo tài sản (asset) trong luồng runtime.

---

### [P2] Interface cho strategy trong tài liệu/template đã cũ so với code

**Bằng chứng**

- Lớp Base thực tế sử dụng:
  - `supported_asset_types`, `supported_regimes`
  - `synth/miner/strategies/base.py:67-69`
- Tài liệu + template vẫn hướng dẫn sử dụng:
  - `supported_assets`, `supported_frequencies`
  - `docs/STRATEGY_SPEC.md:10-11`
  - `docs/templates/strategy_template.py:14-15`
  - `docs/DEVELOPER_ONBOARDING_GUIDE.md:88-89`

**Tác động**

- Lập trình viên mới dễ thiết lập sai metadata của strategy -> khiến bộ lọc asset/regime không theo đúng ý.
- Các công cụ (tooling) có sử dụng trường dữ liệu cũ sẽ bị crash (đã gặp trong dòng lệnh duel CLI, xem tài liệu của task 2).

**Khuyến nghị**

- Đồng bộ hóa tài liệu/template theo lớp `BaseStrategy` của hiện tại.
- Thêm cảnh báo tương thích ("compat warning") vào registry nếu strategy có sử dụng trường dữ liệu định dạng cũ.

---

### [P2] Tồn tại 2 trừu tượng hóa (abstraction) để tải dữ liệu song song, quyền sở hữu chưa rõ ràng

**Bằng chứng**

- Có `UnifiedDataLoader` (runtime đang sử dụng):
  - `synth/miner/my_simulation.py:136-162`
- Lại có ngăn xếp (stack) `pipeline/PriceLoader` và `DataProvider` riêng:
  - `synth/miner/pipeline/loader.py:19-37`
  - `synth/miner/pipeline/base.py:12-21`

**Tác động**

- Làm tăng chi phí bảo trì.
- Dễ xảy ra sự sai lệch hành vi (behavior drift) giữa luồng "được xem là mới" và luồng "đang thực sự được chạy".

**Khuyến nghị**

- Chốt sử dụng 1 luồng chính duy nhất.
- Nếu giữ lại thư mục `pipeline/`, cần phải chuyển các điểm gọi thực tế để sử dụng đúng nó và loại bỏ luồng code cũ.

## 4. Bản đồ kiến trúc "hiện trạng" (as-is - thực thi)

1. `neurons/miner.py` nhận request và thực hiện lệnh gọi `synth/miner/entry.py`.
2. `entry.py` lấy thông tin định tuyến (routing) từ `StrategyStore` (`strategies.yaml`) cộng tính năng auto-discovery từ strategy registry.
3. `entry.py` gọi hàm `simulate_crypto_price_paths(...)`.
4. `simulate_crypto_price_paths` tiến hành lấy dữ liệu thông qua `UnifiedDataLoader` (hiện tại thì đang phân rã khung thời gian dựa trên các nhãn asset).
5. Strategy sinh ra đường giá (path) -> đánh giá điểm đầu ra (validate output) -> áp dụng theo các kết quả luồng tính dự phòng (fallback chain) nếu cần.
6. Hệ backtest runner sử dụng chức năng trong DataHandler dùng để bóc tách thông tin history, và thư viện `cal_reward` (cũ) được dùng cho mục đích tính điểm CRPS.

## 5. Đề xuất lộ trình kiến trúc

### Pha A

- Đồng bộ hoá toàn bộ tài liệu hợp đồng của strategy.
- Cấu trúc lại dataloader nhằm nhận tham số frequency/time_increment rành mạch rõ ràng.

### Pha B

- Thống nhất duy nhất một tầng phục vụ dữ liệu (ưu tiên MySQL hoặc DuckDB) và cập nhật tài liệu kiến trúc tương ứng đi kèm.
- Bổ sung nhóm integration test phục vụ luồng xử lý low/high để dùng đúng dữ liệu về tính phân giải thời gian.

### Pha C

- Đơn giản hóa kiến trúc dữ liệu data pipeline (xóa đi những khoảng abstraction bị dư/không được gọi tới bao giờ).
- Lắp đặt công cụ theo dõi (logging và metric canary) tạo màng phát hiện lỗi trôi dạt (drift) giữa hai bộ backtest và runtime.

## 6. Mức độ tin cậy của đánh giá

- Đã xác thực cẩn thận bằng các dòng lệnh chạy runtime (qua môi trường conda env) cho mọi đầu lỗi hành vi với các nguyên nhân chính của hệ thống tooling.
- Các diễn ngôn kết luận phát hiện kiến trúc đều đi kèm đánh dấu mốc dòng (line-level reference) thực thụ tồn tại trong codebase hiện thời và tài liệu có sẵn.
