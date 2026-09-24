from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


TIMEZONE = "Asia/Ho_Chi_Minh"
DEFAULT_LAT = 20.95
DEFAULT_LON = 107.08


def build_default_forecast_dates(days=7):
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    return [today + timedelta(days=i) for i in range(days)]


def _normalize_expected_dates(forecast_dates):
    dates = pd.to_datetime(list(forecast_dates)).normalize()

    if len(dates) != 7:
        raise ValueError("forecast_dates phải chứa đúng 7 ngày.")

    if dates.duplicated().any():
        raise ValueError("forecast_dates không được chứa ngày trùng lặp.")

    expected = pd.date_range(dates[0], periods=7, freq="D")
    if not dates.equals(expected):
        raise ValueError("forecast_dates phải là 7 ngày liên tiếp theo thứ tự tăng dần.")

    return dates


def get_real_weather_7_days(
    forecast_dates=None,
    lat=DEFAULT_LAT,
    lon=DEFAULT_LON,
    save_path=None
):
    """
    Lấy dự báo thời tiết Open-Meteo và ép kết quả khớp chính xác
    với canonical 7-day horizon do main_pipeline cung cấp.

    Schema trả về được chuẩn hóa để dùng trực tiếp cho weather risk model:
    Ngay, Nhiet_Do_C, Luong_Mua_mm, Suc_Gio_kmh, Bao.
    """
    if forecast_dates is None:
        forecast_dates = build_default_forecast_dates()

    expected_dates = _normalize_expected_dates(forecast_dates)

    print(
        f"📡 Đang tải thời tiết tại (Lat: {lat}, Lon: {lon}) "
        f"cho {expected_dates[0].date()} → {expected_dates[-1].date()}..."
    )

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,rain_sum,wind_speed_10m_max",
        "timezone": TIMEZONE,
        "forecast_days": 7,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        daily = response.json()["daily"]

        weather_df = pd.DataFrame({
            "Ngay": pd.to_datetime(daily["time"]).normalize(),
            "Nhiet_Do_C": np.round(daily["temperature_2m_max"], 1),
            "Luong_Mua_mm": np.round(daily["rain_sum"], 1),
            "Suc_Gio_kmh": np.round(daily["wind_speed_10m_max"], 1),
        })

        weather_df["Bao"] = (
            (weather_df["Luong_Mua_mm"] > 100)
            & (weather_df["Suc_Gio_kmh"] > 50)
        ).astype(int)

        # Chỉ giữ đúng canonical horizon và xác thực 1:1 theo ngày.
        weather_df = (
            pd.DataFrame({"Ngay": expected_dates})
            .merge(weather_df, on="Ngay", how="left", validate="one_to_one")
        )

        missing_dates = weather_df.loc[
            weather_df[["Nhiet_Do_C", "Luong_Mua_mm", "Suc_Gio_kmh"]]
            .isna()
            .any(axis=1),
            "Ngay"
        ]

        if not missing_dates.empty:
            missing_text = ", ".join(d.strftime("%Y-%m-%d") for d in missing_dates)
            raise ValueError(
                "Open-Meteo không trả đủ dữ liệu cho canonical horizon. "
                f"Thiếu: {missing_text}"
            )

        weather_df["Ngay"] = weather_df["Ngay"].dt.strftime("%Y-%m-%d")

        if save_path:
            weather_df.to_csv(save_path, index=False)
            print(f"✅ Đã lưu weather forecast: '{save_path}'")

        print("✅ Dữ liệu thời tiết đã khớp chính xác với 7 ngày dự báo nhu cầu.")
        print(weather_df.to_string(index=False))

        return weather_df

    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        print(f"❌ Lỗi tải/kiểm tra dữ liệu thời tiết: {exc}")
        return None


if __name__ == "__main__":
    get_real_weather_7_days(save_path="Weather_Forecast_7days.csv")
