import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from fetch_real_weather import get_real_weather_7_days
from layer1_machinelearning import load_and_preprocess, run_demand_forecast


TIMEZONE = "Asia/Ho_Chi_Minh"
FORECAST_DAYS = 7
DEMAND_OUTPUT = "D_t_forecast_7days.csv"
WEATHER_OUTPUT = "Weather_Forecast_7days.csv"
PIPELINE_CONTEXT = "pipeline_context.json"


def run_script(script_name, command_type="python"):
    """
    Thực thi script phụ. Các bước cần chia sẻ forecast horizon được gọi trực tiếp
    bằng Python function thay vì subprocess.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ ĐANG KHỞI CHẠY: {script_name}...")
    try:
        if command_type == "python":
            subprocess.run([sys.executable, script_name], check=True)
        elif command_type == "streamlit":
            print(
                f"[{time.strftime('%H:%M:%S')}] 🌐 "
                "Đang khởi động Máy chủ Giao diện Web (Dashboard)..."
            )
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_name])
            return True
        else:
            raise ValueError(f"command_type không hợp lệ: {command_type}")

        print(f"[{time.strftime('%H:%M:%S')}] ✅ HOÀN THÀNH: {script_name}")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        print(f"\n❌ LỖI khi chạy '{script_name}': {exc}")
        return False


def build_forecast_horizon(days=FORECAST_DAYS):
    """
    Canonical horizon duy nhất của toàn hệ thống.
    Với chế độ real-time: hôm nay -> hôm nay + 6 theo giờ Việt Nam.
    """
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    return [today + timedelta(days=i) for i in range(days)]


def validate_demand_freshness(data_path, forecast_dates):
    """
    Real-time forecast yêu cầu dữ liệu nhu cầu cập nhật ít nhất đến ngày hôm qua.

    Để chạy demo/simulation với dữ liệu lịch sử cũ, có thể chủ động đặt:
        ALLOW_STALE_DEMAND_DATA=1

    Việc override được ghi rõ ra console để tránh nhầm dữ liệu cũ là real-time.
    """
    df = load_and_preprocess(data_path)
    if df is None or df.empty:
        raise RuntimeError(f"Không đọc được dữ liệu từ '{data_path}'.")

    last_observation = pd.Timestamp(df["Ngay"].max()).date()
    required_last_date = forecast_dates[0] - timedelta(days=1)

    if last_observation == required_last_date:
        print(
            f"✅ Demand data freshness OK: dữ liệu cập nhật đến {last_observation}."
        )
        return

    message = (
        "Demand data không đủ mới cho REAL-TIME mode. "
        f"Ngày dữ liệu cuối: {last_observation}; "
        f"cần: {required_last_date}."
    )

    allow_stale = os.getenv("ALLOW_STALE_DEMAND_DATA", "0") == "1"
    if allow_stale:
        print(f"⚠️ {message}")
        print(
            "⚠️ ALLOW_STALE_DEMAND_DATA=1 đang bật: tiếp tục ở chế độ "
            "DEMO/SIMULATION. Không diễn giải kết quả là forecast real-time."
        )
        return

    raise RuntimeError(
        message
        + " Hãy cập nhật Data.csv hoặc đặt ALLOW_STALE_DEMAND_DATA=1 "
        "nếu chủ động chạy demo/simulation."
    )


def validate_output_alignment(
    forecast_dates,
    demand_path=DEMAND_OUTPUT,
    weather_path=WEATHER_OUTPUT
):
    """
    Fail-fast invariant:
      - Demand Forecast có đúng 7 ngày canonical.
      - Weather Forecast có đúng 7 ngày canonical.
      - Hai tập ngày bằng nhau và cùng thứ tự.
    """
    expected = pd.DatetimeIndex(pd.to_datetime(forecast_dates)).normalize()

    demand = pd.read_csv(demand_path)
    weather = pd.read_csv(weather_path)

    demand_dates = pd.DatetimeIndex(pd.to_datetime(demand["Ngay"])).normalize()
    weather_dates = pd.DatetimeIndex(pd.to_datetime(weather["Ngay"])).normalize()

    if len(demand_dates) != FORECAST_DAYS or len(weather_dates) != FORECAST_DAYS:
        raise RuntimeError(
            "Demand/Weather output phải chứa đúng 7 dòng tương ứng 7 ngày."
        )

    if not demand_dates.equals(expected):
        raise RuntimeError(
            "D_t_forecast_7days.csv không khớp canonical forecast horizon."
        )

    if not weather_dates.equals(expected):
        raise RuntimeError(
            "Weather_Forecast_7days.csv không khớp canonical forecast horizon."
        )

    if not demand_dates.equals(weather_dates):
        raise RuntimeError(
            "Demand Forecast và Weather Forecast không cùng khoảng ngày."
        )

    print(
        "✅ ALIGNMENT CHECK PASSED: Demand và Weather cùng horizon "
        f"{expected[0].date()} → {expected[-1].date()}."
    )


def write_pipeline_context(forecast_dates):
    now = datetime.now(ZoneInfo(TIMEZONE))
    context = {
        "generated_at": now.isoformat(),
        "timezone": TIMEZONE,
        "forecast_start": forecast_dates[0].isoformat(),
        "forecast_end": forecast_dates[-1].isoformat(),
        "forecast_days": len(forecast_dates),
        "allow_stale_demand_data": os.getenv(
            "ALLOW_STALE_DEMAND_DATA", "0"
        ) == "1",
    }

    with open(PIPELINE_CONTEXT, "w", encoding="utf-8") as file:
        json.dump(context, file, ensure_ascii=False, indent=2)


def orchestrate_pipeline():
    """
    End-to-End Pipeline với một canonical 7-day horizon dùng chung
    cho Demand Forecast và Weather Forecast.
    """
    print("=" * 72)
    print("🚀 AI SUPPLY CHAIN DSS - ALIGNED 7-DAY PIPELINE")
    print("=" * 72)

    forecast_dates = build_forecast_horizon()
    print(
        "📅 Canonical Decision Horizon: "
        f"{forecast_dates[0]} → {forecast_dates[-1]}"
    )

    try:
        # BƯỚC 1: Kiểm tra dữ liệu demand có phù hợp real-time hay không.
        validate_demand_freshness("Data.csv", forecast_dates)

        # BƯỚC 2: Chuẩn bị dữ liệu/model Weather Risk AI.
        if not os.path.exists("Weather_Delivery_History.csv"):
            print(
                "\n[HỆ THỐNG] Chưa có Weather_Delivery_History.csv. "
                "Đang tạo dữ liệu mô phỏng để huấn luyện risk model..."
            )
            if not run_script("generate_weather_data.py"):
                return False

        if not os.path.exists("weather_risk_model.pkl"):
            print(
                "\n[HỆ THỐNG] Chưa có weather_risk_model.pkl. "
                "Đang huấn luyện Lớp 1.5..."
            )
            if not run_script("layer1_5_weather_risk.py"):
                return False
        else:
            print("✅ Weather Risk model đã sẵn sàng.")

        # BƯỚC 3: Demand Forecast dùng đúng canonical horizon.
        print("\n[HỆ THỐNG] Chạy Demand Forecast cho canonical horizon...")
        run_demand_forecast(
            data_path="Data.csv",
            output_path=DEMAND_OUTPUT,
            forecast_dates=forecast_dates,
        )

        # BƯỚC 4: Weather Forecast dùng cùng canonical horizon.
        print("\n[HỆ THỐNG] Tải Weather Forecast cho cùng canonical horizon...")
        weather_df = get_real_weather_7_days(
            forecast_dates=forecast_dates,
            save_path=WEATHER_OUTPUT,
        )
        if weather_df is None:
            raise RuntimeError(
                "Không tải được weather forecast; pipeline dừng để tránh ghép sai thời gian."
            )

        # BƯỚC 5: Invariant bắt buộc trước Fuzzy/MILP.
        validate_output_alignment(forecast_dates)
        write_pipeline_context(forecast_dates)

    except Exception as exc:
        print("\n" + "=" * 72)
        print(f"❌ PIPELINE STOPPED: {exc}")
        print("=" * 72)
        return False

    # BƯỚC 6: Chỉ mở Dashboard sau khi Demand/Weather đã aligned.
    print("\n" + "=" * 72)
    print("🎉 PIPELINE HỢP LỆ. ĐANG MỞ DASHBOARD...")
    print("=" * 72)
    return run_script("app_dashboard.py", command_type="streamlit")


if __name__ == "__main__":
    orchestrate_pipeline()
