import pandas as pd
from typing import Optional, Dict
class REGIME_TYPE:
    TRENDING = "TRENDING"
    SIDEWAYS = "SIDEWAYS"

def detect_market_regime_with_bbw(prices: pd.Series):
    """
    Phân tích kỹ hơn để quyết định tham số cho XAU.
    Kết hợp Trend (EMA) và Volatility State (BB Width).
    """
    # 1. Tính Bollinger Band Width (Đo độ nén)
    window = 20
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # BB Width tương đối
    bb_width = (upper - lower) / sma
    
    # Lấy giá trị BBW cuối cùng và trung bình 50 nến gần nhất để so sánh
    current_bbw = bb_width.iloc[-1]
    avg_bbw = bb_width.rolling(50).mean().iloc[-1]
    
    # 2. Xác định Volatility Regime
    # Nếu dải BB đang co lại nhỏ hơn trung bình -> Low Volatility (Squeeze)
    is_squeeze = current_bbw < (avg_bbw * 0.9)
    
    # 3. Xác định Trend (Dùng EMA Slope đơn giản)
    ema_short = prices.ewm(span=10).mean()
    ema_long = prices.ewm(span=30).mean()
    
    # Trend strength
    trend_diff = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / prices.iloc[-1]
    is_trending = trend_diff > 0.002 # Ngưỡng 0.2%
    
    return {
        "is_squeeze": is_squeeze,   # Thị trường đang nén/đi ngang biên độ hẹp
        "is_trending": is_trending, # Có xu hướng rõ ràng
        "bbw_ratio": current_bbw / avg_bbw # Tỷ lệ nén
    }

def detect_market_regime_with_er(prices: pd.Series, lookback: Optional[int] = 20) -> Dict:
    """
    Sử dụng Efficiency Ratio để phát hiện Regime.
    """
    if not lookback:
        lookback = int(len(prices))
    print(f"lookback: {lookback}")
    # 1. Tính Change (Hiệu số giá đầu cuối)
    change = prices.diff(lookback).abs()
    
    # 2. Tính Volatility (Tổng các bước nhảy từng nến)
    volatility = prices.diff().abs().rolling(window=lookback).sum()
    
    # 3. Efficiency Ratio (ER)
    # ER càng gần 1 -> Trend mạnh. ER càng gần 0 -> Sideways/Nhiễu
    er = change / volatility
    current_er = er.iloc[-1]
    
    # 4. Xác định Regime
    # Ngưỡng 0.3 là mốc kinh nghiệm phân tách Trend/Noise
    if current_er > 0.30:
        return {"type": REGIME_TYPE.TRENDING, "strength": current_er}
    else:
        return {"type": REGIME_TYPE.SIDEWAYS, "strength": current_er}

# def detect_market_regime(prices: pd.Series):
#     """
#     Xác định thị trường đang có xu hướng (Trend) hay đi ngang (Range).
#     Trả về: 'trending' hoặc 'sideways'
#     """
#     # Tính EMA ngắn và dài
#     ema_fast = prices.ewm(span=20).mean()
#     ema_slow = prices.ewm(span=50).mean()
    
#     # Tính độ lệch chuẩn của lợi suất (volatility hiện tại)
#     returns = np.log(prices).diff().dropna()
#     vol = returns.std()
    
#     # Tính khoảng cách giữa 2 EMA theo đơn vị volatility
#     dist = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / prices.iloc[-1]
    
#     # Nếu đường trung bình tách nhau đủ xa -> Trending
#     if dist > 0.005: # Ngưỡng 0.5% (có thể điều chỉnh tùy asset)
#         return "trending"
#     else:
#         return "sideways"