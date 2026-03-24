#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Thêm đường dẫn gốc để có thể import từ synth
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from synth.miner.mysql_handler import MySQLHandler

def main(data_type: str, asset: str, start: str, end: str, out: str = None):
    """
    Vẽ biểu đồ giá từ database.
    
    Args:
        data_type: "high" (time_frame 1m) hoặc "low" (time_frame 5m)
        asset: Tên token, ví dụ "BTC", "ETH"
        start: Thời gian bắt đầu (YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS) theo UTC
        end: Thời gian kết thúc (YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS) theo UTC
        out: Đường dẫn file ảnh đầu ra (mặc định sẽ tự động tạo tên file)
    """
    if data_type not in ["high", "low"]:
        raise ValueError("data_type phải là 'high' hoặc 'low'")

    # Hàm hỗ trợ parse date
    def parse_date(d_str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(d_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ValueError(f"Sai định dạng ngày tháng (yêu cầu YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS): {d_str}")

    try:
        start_dt = parse_date(start)
        end_dt = parse_date(end)
    except ValueError as e:
        print(f"Lỗi: {e}")
        return

    # Luôn lấy data 1m từ DB để biến đổi nếu cần
    db_time_frame = "1m"
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print(f"Đang tải {db_time_frame} data cho {asset} từ {start_dt} đến {end_dt} ...")

    mysql = MySQLHandler()
    table_name = mysql._get_table_name(asset)

    try:
        conn = mysql._get_connection()
        try:
            with conn.cursor() as cur:
                query = f"""
                    SELECT timestamp, price FROM {table_name}
                    WHERE time_frame = %s AND timestamp >= %s AND timestamp <= %s
                    ORDER BY timestamp ASC
                """
                cur.execute(query, (db_time_frame, start_ts, end_ts))
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"Lỗi database khi truy vấn bảng {table_name}: {e}")
        return

    if not rows:
        print(f"Không tìm thấy dữ liệu giá 1m nào cho {asset} trong khoảng thời gian đã nhập.")
        return

    timestamps = [int(row[0]) for row in rows]
    prices = [float(row[1]) for row in rows]
    
    if data_type == "low":
        print(f"Đang tổng hợp dữ liệu từ 1m sang 5m cho {len(rows)} điểm nến 1m...")
        from synth.miner.price_aggregation import aggregate_1m_to_5m
        
        # Chuyển cấu trúc [timestamp], [price] thành dictionary mapping
        prices_1m_dict = {str(ts): p for ts, p in zip(timestamps, prices)}
        prices_5m_dict = aggregate_1m_to_5m(prices_1m_dict)
        
        # Trích xuất lại ra list timestamps và prices cho hàm matplotlib plot
        sorted_keys = sorted([int(k) for k in prices_5m_dict.keys()])
        timestamps = sorted_keys
        prices = [prices_5m_dict[str(k)] for k in sorted_keys]

    # Chuyển đổi timestamp sang object datetime để matplotlib vẽ mượt hơn
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]

    out_file = out
    if not out_file:
        time_str = start_dt.strftime("%Y%m%d")
        out_file = f"price_plot_{asset}_{data_type}_{time_str}.png"

    # Bước vẽ Plot
    plt.figure(figsize=(14, 7))
    plt.plot(dates, prices, color='#ff7f0e' if data_type == "low" else '#1f77b4', linewidth=1.5, label=f"Giá {asset}")
    
    # Fill màu nhạt ở dưới đường giá
    plt.fill_between(dates, prices, [min(prices) * 0.999]*len(prices), 
                     color='#ff7f0e' if data_type == "low" else '#1f77b4', alpha=0.1)
    
    time_frame_label = "5m (aggregated)" if data_type == "low" else "1m"
    plt.title(f"Đường giá {asset} (Loại: {data_type.upper()} | {time_frame_label})\nTừ {start} đến {end}", fontsize=14, fontweight='bold')
    plt.xlabel("Thời gian (UTC)", fontsize=12)
    plt.ylabel("Giá", fontsize=12)
    
    # Format hiển thị trục x cho dễ nhìn
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=150)
    print(f"Đã vẽ thành công {len(dates)} điểm dữ liệu.")
    print(f"Biểu đồ đã được lưu tại: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    # ĐIỀN THAM SỐ VÀO ĐÂY VÀ CHẠY FILE
    DATA_TYPE = "high"             # "high" (dùng time_frame 1m) hoặc "low" (dùng time_frame 5m)
    ASSET = "ETH"                  # Đồng/Tài sản (VD: BTC, ETH, SPYX)
    START_DATE = "2026-03-10"      # Thời gian Bắt đầu (YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS)
    END_DATE = "2026-03-24"        # Thời gian Kết thúc (YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS)
    OUT_FILE = None                # Tên file ảnh ra. Nếu để None, sẽ tự đặt rên tên theo asset và thời gian
    
    main(DATA_TYPE, ASSET, START_DATE, END_DATE, OUT_FILE)

