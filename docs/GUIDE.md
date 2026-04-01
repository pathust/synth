**Hướng dẫn chi tiết API SynthData (dựa trên docs tại link bạn cung cấp)**

Base URL chung:  
`https://api.synthdata.co`

**Không cần authentication** (không cần API key, tất cả endpoint đều public).

Dưới đây là các **endpoint phù hợp nhất để fetch data phục vụ backtest & so sánh** (đặc biệt là validation scores lịch sử của các miner). Đây chính là dữ liệu dùng để đánh giá hiệu suất dự đoán (CRPS score, prompt_score), so sánh giữa các miner/asset/thời gian – rất lý tưởng cho backtesting chiến lược hoặc model.

### 1. Endpoint chính (bạn đang tham khảo): Historical Validation Scores
**GET** `/validation/scores/historical`

**Mô tả**: Lấy điểm validation scores của tất cả miner trong khoảng thời gian chỉ định. Đây là endpoint **tốt nhất cho backtest & so sánh** vì trả về dữ liệu lịch sử chi tiết (CRPS, prompt_score, thời gian).

**Query Parameters** (tất cả đều là query string):
- `from` (required, string): Ngày bắt đầu (định dạng ISO 8601).  
  Ví dụ: `2025-02-03`, `2025-02-03T10:19:04Z`, `2025-02-03T17:19:04+07:00`
- `to` (required, string): Ngày kết thúc (cùng định dạng với `from`).
- `miner_uid` (optional, integer): Lọc theo miner cụ thể (nếu không truyền sẽ lấy tất cả).
- `asset` (optional, string): Mã tài sản (mặc định `BTC`).  
  Giá trị cho phép: `BTC`, `ETH`, `XAU`, `SOL`, `SPYX`, `NVDAX`, `TSLAX`, `AAPLX`, `GOOGLX`
- `time_increment` (optional, integer): Khoảng cách giữa các prompt (giây).  
  Chỉ nhận `300` hoặc `60` (mặc định `300`).
- `time_length` (optional, integer): Độ dài cửa sổ prompt (giây).  
  Chỉ nhận `86400` (24h) hoặc `3600` (1h) (mặc định `86400`).

**Ví dụ URL đầy đủ** (backtest BTC 24h, từ 1/3/2025 đến 5/3/2025):
```
https://api.synthdata.co/validation/scores/historical?from=2025-03-01&to=2025-03-05&asset=BTC&time_increment=300&time_length=86400
```

**Ví dụ URL lọc 1 miner**:
```
https://api.synthdata.co/validation/scores/historical?from=2025-03-01&to=2025-03-05&miner_uid=123&asset=BTC
```

**Response (200 OK)** – mảng JSON:
```json
[
  {
    "asset": "BTC",
    "crps": 0.0123,
    "miner_uid": 123,
    "prompt_score": 0.85,
    "scored_time": "2025-03-01T10:00:00Z",
    "time_length": 86400
  },
  ...
]
```

### 2. Latest Validation Scores (dùng để so sánh realtime hoặc snapshot mới nhất)
**GET** `/validation/scores/latest`

**Query Parameters** (không cần from/to):
- `asset` (optional, string): mặc định `BTC` (cùng danh sách như trên).
- `time_increment` (optional): `300` hoặc `60` (mặc định 300).
- `time_length` (optional): `86400` hoặc `3600` (mặc định 86400).

**Ví dụ**:
```
https://api.synthdata.co/validation/scores/latest?asset=BTC&time_length=3600
```

**Response**: Tương tự endpoint historical nhưng chỉ lấy điểm mới nhất.

### 3. Các endpoint bổ sung hữu ích cho so sánh/backtest (Leaderboard)
Nếu bạn muốn so sánh **xếp hạng tổng thể** hoặc **meta-leaderboard** (tổng hợp reward dài hạn):

- **GET** `/v2/leaderboard/historical`  
  (lấy lịch sử leaderboard)  
  Params: `start_time` (required), `end_time` (required), `prompt_name` (`high` hoặc `low`).

- **GET** `/v2/meta-leaderboard/historical`  
  (tổng hợp reward theo số ngày)  
  Params: `start_time` (required), `prompt_name`, `days` (mặc định 14).

Ví dụ:
```
https://api.synthdata.co/v2/meta-leaderboard/historical?start_time=2025-02-01&days=30&prompt_name=low
```

### Hướng dẫn sử dụng cho Backtest & So sánh
1. Gọi `/validation/scores/historical` với khoảng thời gian bạn muốn backtest.
2. Lưu kết quả vào CSV/database.
3. So sánh:
   - Theo `miner_uid` (ai có CRPS thấp hơn = dự đoán chính xác hơn).
   - Theo `asset` (BTC vs ETH…).
   - Theo `prompt_score` hoặc `time_length`.
4. Kết hợp với `/validation/scores/latest` để kiểm tra hiệu suất realtime.

**Lưu ý**:
- Tất cả response đều là JSON.
- Không có rate-limit được ghi trong docs (nhưng nên gọi có khoảng cách).
- Nếu muốn backtest dài hạn → chia nhỏ request (ví dụ mỗi lần 30 ngày).

Bạn chỉ cần copy-paste URL ví dụ là có thể fetch ngay bằng Postman, Python `requests`, hoặc code backtest của bạn. Nếu cần code mẫu Python để fetch + lưu CSV, cứ bảo mình nhé!