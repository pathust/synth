# Strategy Specification

## BaseStrategy Interface
Đây là "hợp đồng" (contract) mà mọi strategy phải tuân thủ. Không sử dụng global state.

```python
class BaseStrategy(ABC):
    name: str = ""
    version: str = "1.0"
    description: str = ""
    supported_asset_types: list[str] = []
    supported_regimes: list[str] = []
    default_params: dict = {}
    param_grid: dict = {}
    
    @abstractmethod
    def simulate(
        self,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        n_sims: int,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> np.ndarray:
        """
        Thực thi mô phỏng giá.
        - prices_dict: Dữ liệu giá lịch sử.
        - asset: Tên tài sản (VD: BTC).
        - time_increment: Bước thời gian (giây).
        - time_length: Tổng thời gian mô phỏng (giây).
        - n_sims: Số lượng đường (paths).
        - seed: Seed để tái lập kết quả.
        Trả về: np.ndarray shape (n_sims, steps + 1)
        """
        pass
```

## StrategyRegistry Design
Thiết kế mới thay thế biến global `_registry` bằng instance-based registry, kết hợp Dependency Injection (DI).

- **Instance-based**: `registry = StrategyRegistry()` được khởi tạo tại entry point (miner/backtest).
- **DI**: Truyền `registry` vào các component cần thiết thay vì import global.
- **Lookup**: `registry.get(name)` hoặc `registry.get_for_asset(asset, regime)`.
- **Reset trong test**: Đơn giản tạo một instance `StrategyRegistry` mới cho mỗi test case (sử dụng pytest fixture). Không còn hiện tượng test pollution.

## Strategy Inventory Table

| Strategy Name | File Path | Market | Timeframe | Core Logic | Known Issues | Action |
|---------------|-----------|--------|-----------|------------|--------------|--------|
| `garch_v1` | `strategies/garch_v1.py` | Crypto | HFT/LFT | GARCH(1,1) cơ bản | Kém linh hoạt với volatility spike | Deprecate |
| `garch_v2` | `strategies/garch_v2.py` | Crypto | HFT/LFT | Cải tiến từ v1 | Trùng lặp code với v1 | Refactor (gom chung thành `garch`) |
| `gjr_garch` | `strategies/gjr_garch.py` | Crypto/Equity | LFT | Bắt dính leverage effect | Tính toán chậm do optimize | Keep |
| `egarch` | `strategies/egarch.py` | Equity | LFT | Xử lý asymmetric volatility | Không hội tụ trong một số điều kiện | Keep |
| `jump_diffusion` | `strategies/jump_diffusion.py` | Crypto | HFT | Thêm tham số nhảy giá (Merton) | Thiếu kiểm soát đuôi phân phối | Refactor |
| `mean_reversion` | `strategies/mean_reversion.py` | Gold/Forex | LFT | Ornstein-Uhlenbeck | Sai số khi có trend mạnh | Refactor |
| `arima_equity` | `strategies/arima_equity.py` | Equity | LFT | Dự báo mean với ARIMA | Rất chậm khi fit | Deprecate |
| `regime_switching` | `strategies/regime_switching.py` | Crypto | LFT | HMM 2 trạng thái | Dễ overfitting | Refactor |
| `ensemble_weighted` | `strategies/ensemble_weighted.py` | All | All | Trộn các model theo trọng số | Hardcode trọng số | Refactor |
| `pattern_detector` | `strategies/pattern_detector.py` | Equity | HFT | Rule-based (RSI, MACD) | Khó bảo trì, nhiều magic number | Deprecate |

## Refactor Instructions per Strategy

### 1. `garch` (Gom từ garch_v1, garch_v2, garch_v4)
- **Thay đổi để conform**: Đổi logic khởi tạo sang `simulate()` với `**kwargs` hỗ trợ `p, q`.
- **Edge cases**: Handle lỗi hội tụ (convergence warning) bằng cách set default fallback parameters.
- **Tunable params**: `p`, `q`, `mean`, `vol`.

### 2. `jump_diffusion`
- **Thay đổi để conform**: Tách logic sinh đường chuẩn (Wiener) và sinh bước nhảy (Poisson) để dễ test.
- **Edge cases**: Giá âm (nếu dùng mô hình sai), cần đảm bảo log-normal distribution.
- **Tunable params**: `jump_intensity`, `jump_mean`, `jump_std`.

### 3. `ensemble_weighted`
- **Thay đổi để conform**: Đọc list strategies và trọng số từ `kwargs` thay vì hardcode. 
- **Edge cases**: Tổng trọng số khác 1 (cần chuẩn hóa lại). Trọng số âm (báo lỗi).
- **Tunable params**: List các sub-model và weights.

## New Strategies Proposal

### 1. Heston Stochastic Volatility Model
- **Logic tóm tắt**: Mô phỏng sự biến động của giá bằng hai phương trình vi phân ngẫu nhiên (SDE), một cho giá, một cho phương sai.
- **Lý do đề xuất**: Khắc phục nhược điểm của GARCH trong HFT (GARCH phản ứng chậm với biến động).
- **Thị trường / Timeframe**: Crypto / HFT.
- **Priority**: High.

### 2. LSTM Volatility Predictor
- **Logic tóm tắt**: Dùng mạng LSTM dự báo volatility của N bước tiếp theo, sau đó feed vào phương trình Geometric Brownian Motion.
- **Lý do đề xuất**: Nắm bắt các mẫu phi tuyến tính mà mô hình thống kê bỏ lỡ.
- **Thị trường / Timeframe**: Crypto/Equity / LFT.
- **Priority**: Medium.

### 3. SABR Model
- **Logic tóm tắt**: Mô hình chuyên dụng cho hàng hóa, mô phỏng stochastic volatility.
- **Lý do đề xuất**: Mean-reversion hiện tại chạy không tốt cho vàng (Gold). SABR giải quyết được đường cong volatility smile.
- **Thị trường / Timeframe**: Gold / HFT & LFT.
- **Priority**: Medium.
