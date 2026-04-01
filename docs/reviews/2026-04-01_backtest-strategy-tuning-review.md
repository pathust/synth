# Đánh Giá Chi Tiết: Backtest, Strategy, Tuning

Ngày đánh giá: 2026-04-01  
Người đánh giá: Codex (xác minh mức code và lệnh trong `conda -n synth`)

## 1. Tóm tắt điều hành (Executive summary)

Cụm `backtest/strategy/tuning` đã có khung tổ chức rõ ràng (runner, tuner, regime engine, exporter), nhưng hiện có một số lỗi logic có thể làm sai kết luận của tuning và định tuyến (routing) triển khai:

- Có lỗi về horizon/frequency (chân trời/tần suất) trong pipeline regime backtest.
- Có lỗi sai khác khoảng đánh giá điểm gap (sai khác tên gọi và lỗi chỉ mục).
- Có sự trôi dạt dữ liệu high/low giữa các luồng backtest slot và runtime simulation.
- Có một số lỗi trong các công cụ CLI làm hỏng quy trình làm việc (workflow) kiểm tra nhanh.

## 2. Các phát hiện (xếp hạng theo mức độ nghiêm trọng)

### [P1] Regime backtest đang điều chỉnh/đánh giá theo `config.frequency`, bỏ qua `case.market_type`

**Bằng chứng**

- Trong `evaluate_case`, cả tune/validation/test đều truyền `config.frequency`, không dùng `case_key.market_type`:
  - `synth/miner/backtest/regime_engine.py:455-459`
  - `synth/miner/backtest/regime_engine.py:469-473`
  - `synth/miner/backtest/regime_engine.py:525-529`
- Khi xuất ra cho runtime, trường hợp crypto `market_type=high|low` lại được ánh xạ thành frequency high/low:
  - `synth/miner/backtest/regime_engine.py:729-733`
- Tham số frequency mặc định của CLI là `low`:
  - `synth/miner/run_regime_backtest.py:43`

**Tác động**

- Trường hợp `btc/high/*` có thể được chọn strategy dựa trên thông số backtest low (24h) nhưng vẫn xuất thông tin ra nhóm (bucket) high (1h).
- Tệp định tuyến (routing YAML) sinh ra để deploy có nguy cơ "luồng high được tune bằng horizon low".

**Khuyến nghị**

- Tính toán `effective_frequency` theo `case_key.market_type` (crypto high/low), không sử dụng duy nhất một biến toàn cục cho mọi trường hợp.
- Bổ sung test assert kiểm tra frequency theo trường hợp cụ thể.

---

### [P1] Quãng tính điểm "gap" của HFT không được kích hoạt đúng

**Bằng chứng**

- Trong cấu hình prompt (prompt config), tên của khoảng (interval) có dạng là `_gaps`:
  - `synth/validator/prompt_config.py:66-77`
- CRPS logic lại kiểm tra với hậu tố `_gap`:
  - `synth/validator/crps_calculation.py:37-39`
  - `synth/miner/backtest/metrics.py:56-58`
- Xác minh bằng lệnh thực tế:
  - `conda run -n synth python -c "from synth.validator.prompt_config import HIGH_FREQUENCY; ..."`
  - Kết quả: `0` khóa có đuôi `_gap`, `12` khóa có đuôi `_gaps`.

**Tác động**

- Toàn bộ khối dùng tính điểm gap trong HFT không bao giờ được bật như ý định ban đầu.
- CRPS thực tế bị sai khác so với thiết kế tính điểm trong prompt config.

**Khuyến nghị**

- Chốt sử dụng một quy ước đặt tên chung (`_gap` hoặc `_gaps`) và đồng bộ giữa prompt config và module tính điểm.
- Viết thêm unit test để đảm bảo phần mềm đọc hậu tố cho đúng.

---

### [P1] Nhánh gap mode đang có mầm mống lỗi chỉ mục (index) nếu thực sự được kích hoạt

**Bằng chứng**

- Khi `is_gap=True`, mã nguồn thực hiện cắt mảng `interval_prices = interval_prices[:1]`:
  - `synth/validator/crps_calculation.py:167-170`
- Xác minh bằng lệnh:
  - `conda run -n synth python -c "... calculate_price_changes_over_intervals(..., is_gap=True) ..."`
  - Output có shape `(1, 3)` cho danh sách 2 path đầu vào. Nó lấy vị trí 1 => làm mất toàn bộ các đường path còn lại, chứ không phải lấy "1 gap trải dọc theo chiều thời gian".

**Tác động**

- Nếu khắc phục xong lỗi chọn hậu tố, điều này sẽ làm sụp hẳn nhánh xử lý và tạo điểm số sai lệch hoàn toàn.

**Khuyến nghị**

- Định nghĩa rõ "gap interval" phải có định dạng kết quả đầu ra như thế nào, rồi sửa phép cắt khối (slicing) theo chiều của thời gian (chứ không phải theo chỉ mục path).
- Củng cố bằng một bài regression test cho shape và giá trị cuối cùng.

---

### [P1] Cả backtest slot và runtime simulation đang "ưu tiên luồng high" đối với các tài sản có frequency đôi (dual-frequency)

**Bằng chứng**

- `BacktestEngine._get_prompt_config` ưu tiên gán nhánh HIGH nếu tài sản cho dán cả bộ nhãn đó:
  - `synth/miner/backtest/engine.py:70-74`
- Thao tác gộp `run_slots` hiện tại từ chối tham số argument cho phép; nó nhặt cfg từ trong chi tiết của asset:
  - `synth/miner/backtest/engine.py:87-95`
- `UnifiedDataLoader` cũng có cách tìm kiếm mapping dọc theo nhãn (labels):
  - `synth/miner/data/dataloader.py:22-26`
- Kiểm tra lại:
  - `conda run -n synth python -c "from synth.miner.backtest.engine import BacktestEngine; ..."`
  - Hệ thống cho ra: `BTC -> high`, `NVDAX -> low`.

**Tác động**

- Gây khó hoặc vô hiệu hóa việc ước lượng trên dòng slot của low cho BTC/ETH/SOL/XAU khi chạy `run_slots`.
- Nguy cơ tải khung granularity của 1m lên cho 5m ở horizon thấp.

**Khuyến nghị**

- Ràng buộc tham số truyền vào (explicit argument) cho `frequency`/`time_increment` khi làm việc cùng đối tượng `run_slots` và khi thiết kế các APIs gọi dataloader.

---

### [P2] Lệnh bash `run_strategy_scan --metric DIR_ACC` bị lỗi ngưng ngay lúc startup

**Bằng chứng**

- Công cụ CLI hỗ trợ cho phép truyền vào metric `DIR_ACC`:
  - `synth/miner/run_strategy_scan.py:84-87`
- Tuy nhiên khối Runner chỉ chấp thuận danh sách metric nhất định thuộc `METRICS`:
  - `synth/miner/backtest/runner.py:40-43`
  - `synth/miner/backtest/metrics.py:170-174`
- Xác thực:
  - `conda run -n synth python -m synth.miner.run_strategy_scan --assets BTC --frequencies low --num-runs 1 --num-sims 1 --metric DIR_ACC`
  - Lỗi sinh ra: `ValueError: Unknown metric 'DIR_ACC'. Available: ['CRPS', 'MAE', 'RMSE']`.

**Tác động**

- Hợp đồng của CLI sai, báo lỗi gây phiền cho quá trình dùng tay và chặn đứng script chạy CI tự động.

**Khuyến nghị**

- Hoặc bỏ tùy chọn `DIR_ACC` khỏi menu, hoặc viết bù đúng đoạn mã tính toán metric vào `METRICS`.

---

### [P2] Tính năng Backtest low-frequency có khả năng lỗi dù DB đang chứa dữ liệu 1m (do phớt lờ thao tác lưu của bước hợp nhất - aggregate)

**Bằng chứng**

- Lệnh gọi `fetch_price_data(..., only_load=True)` chuyển quy trình chạy nhánh gộp dữ liệu 1m biến thành 5m rồi return lại trực tiếp kết quả:
  - `synth/miner/my_simulation.py:67-83`
- Vấn đề là `run_single` gọi xong lại bỏ vào ngăn trống không lưu bộ nhớ output này; tiếp theo lại vội vàng query để đòi đúng dữ liệu gốc 5m từ DB:
  - `synth/miner/backtest/runner.py:75-80`

**Tác động**

- Nếu rỗng trong DB dòng của 5m nhưng thỏa 1m, quá trình chạy của backtest vẫn chết bằng thông điệp FAIL "No data for asset/5m".

**Khuyến nghị**

- Yêu cầu khai thác data được chuyển về từ bước `fetch_price_data`, hoặc bắt buộc ghi xuống disk cho nhóm nén gộp 5m lúc vừa gọi hàm `load_price_data`.

---

### [P2] Nguồn mầm ngẫu nhiên (seed) của phép lấy mẫu tổng hợp (ensemble) không tái lập lại được (reproducible) giữa nhiều session độc lập (process)

**Bằng chứng**

- Sub-seed đang dùng lời gọi có sẵn Python `hash(...)`:
  - `synth/miner/ensemble/builder.py:18-20`
  - `synth/miner/strategies/ensemble_weighted.py:97`
- Tiến hành thực chứng trên 2 quá trình chạy độc lập vừa thao tác liền tiếp nhau:
  - Lần 1: `_make_sub_seed(42,'garch_v4') -> 82778`
  - Lần 2: `_make_sub_seed(42,'garch_v4') -> 82248`

**Tác động**

- Set đầy đủ thông số giống y xì nhưng giá trị seed ngẫu nhiên lại lệch hướng liên tục cứ mỗi lần hệ thống bị reload.
- Hao mòn độ chất lượng cũng như tính chính xác công tâm lúc rà soát thông số tuning.

**Khuyến nghị**

- Lắp thuật toán hàm băm an toàn/tĩnh như `hashlib.sha256(strategy_name.encode())` thay thế làm khung sinh seed.

---

### [P3] Lệnh bash `run_duel --list` chết do vấp trúng Metadata cũ kỹ

**Bằng chứng**

- Biến đọc lệnh CLI lục tìm tới các từ khóa nội chuẩn cũ ví dụ như `strat.supported_assets` cũng như cào tìm `strat.supported_frequencies`:
  - `synth/miner/backtest/run_duel.py:57-58`
- Khu vực dữ liệu cho BaseStrategy hiện thực không cấy mảng 2 thông tin cũ đó:
  - `synth/miner/strategies/base.py:67-69`
- Thực nghiệm bằng Terminal:
  - `conda run -n synth python -m synth.miner.backtest.run_duel --list`
  - Cảnh báo bắt tại trận: `AttributeError: 'ArimaEquityStrategy' object has no attribute 'supported_assets'`.

**Tác động**

- Một lệnh rất ngắn để soát inventory đang vô năng lượng gây khó kiểm soát.

**Khuyến nghị**

- Cập nhật dòng trỏ CLI điều phối về lấy đúng metadata (hoặc `supported_asset_types`) hay hàm tra cập method `supports_*`.

---

### [P3] Thuật chạy lệnh rà `scan_all` không thể sàng dữ liệu và kiểm cho trùng khớp về tần suất tín hiệu (frequency)

**Bằng chứng**

- Các bước trong `scan_all` rẽ nhánh tạo lưới duyệt `asset x freq` khi nhặt ra từng thành phần chạy strategy đều chỉ lo nhìn vào dòng asset:
  - `synth/miner/backtest/runner.py:277-304`

**Tác động**

- Rơi rớt dư điểm thừa làm cho quét nhầm phải mảng vô phép gây nghẽn làm gia tăng thông báo độ nhiễu/nâng cấp hao phí làm trễ việc.

**Khuyến nghị**

- Thiết lập lệnh rào kiểm bộ lọc `strategy.supports_frequency(freq)` và (hay) một rào soát chéo hỗ trợ chiếu trên không gian support matrix của dữ liệu prompt.

## 3. Các khoảng trống trên độ bao phủ thử nghiệm (Test coverage gaps)

1. Chưa có bộ thử nghiệm (test) cho vụ xử lý bắt nhầm từ khóa đuôi `_gap` trước biến `_gaps` nơi khoảng thông báo thu chấm điểm.
2. Chưa có bộ test mô phỏng phản kháng cho hành vi frequency tùy biến theo từng class của môđun `PredictionBacktestEngine`.
3. Chưa có bộ test kiểm soát hợp quy cho menu bộ thông tin metric gọi truyền từ CLI (`run_strategy_scan`).
4. Chưa chạy thử gói thử nghiệm kiểm chứng độ ổn định dạo ban đầu (smoke) giành chức năng gọi nhanh `run_duel --list`.

## 4. Lộ trình kế hoạch tiến độ xử lý đề xuất

### Nhóm A (Cần giải quyết luôn)

1. Fix lỗi cài chế độ thiết lập tần suất tùy tính hình riêng ở máy thông dịch lõi regime (frequency-per-case).
2. Fix mâu thuẫn khâu băm giải cắt mã định nghĩa cho interval và lát cắt mảng chỉ định thuộc gap slicing.
3. Chỉnh cho chuẩn hợp đồng khai báo (contract) thông tin truy số gọi metric trên lệnh `run_strategy_scan`.

### Nhóm B (Kế tiếp gần)

1. Giao rành rọt tham số thông báo frequency và /hoặc biến bù giờ `time_increment` tới khối dataloader và lệnh `run_slots`.
2. Tạo nguồn hằng định không trượt khóa (hashlib) bảo vệ nguồn cấp hằng sinh chu kỳ (seed) tổng kết ngẫu nhiên.
3. Kịp thời sửa nhầm của hàm `run_duel --list`.

### Nhóm C (Củng cố nền chất lượng chung hệ thống dài lâu)

1. Bổ khuyết loạt chuỗi môđul rà test phối ghép tích hợp cho low và high ở mạch chạy điểm cuối (end-to-end).
2. Quyết đoạt chọn lựa theo đúng chuỗi mạch khai dòng cho chuẩn một path xuyên tâm giành thông luồng từ backtest cho vươn tới phần runtime ngăn rủi ro đứt dòng trượt xa.

## 5. Danh sách tài liệu tra các dòng lệnh thực nghiệm hệ thống

- `conda run -n synth python -m synth.miner.backtest.run_duel --list`
- `conda run -n synth python -m synth.miner.run_strategy_scan --assets BTC --frequencies low --num-runs 1 --num-sims 1 --metric DIR_ACC`
- `conda run -n synth python -c "from synth.miner.backtest.engine import BacktestEngine; ..."`
- `conda run -n synth python -c "from synth.miner.data.dataloader import UnifiedDataLoader; ..."`
- `conda run -n synth python -c "from synth.validator.prompt_config import HIGH_FREQUENCY; ..."`
- `conda run -n synth python -c "import numpy as np; from synth.validator.crps_calculation import calculate_price_changes_over_intervals as f; ..."`
- `conda run -n synth python -c "from synth.miner.ensemble.builder import _make_sub_seed; print(_make_sub_seed(...))"`
