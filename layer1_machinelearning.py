import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess(file_path):
    """
    BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
    Đảm bảo tính toàn vẹn của dữ liệu đầu vào.
    """
    print("⏳ [1/5] Đang tải và làm sạch dữ liệu...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {file_path}.")
        return None

    df['Ngay'] = pd.to_datetime(df['Ngay'], format='mixed', dayfirst=True)
    df = df.sort_values('Ngay').reset_index(drop=True)

    cols_to_fill = [
        'Lo_1_Than', 'Lo_2_Than', 'Lo_1_Chat_Dot', 'Lo_2_Chat_Dot',
        'Lo_1_Carbon', 'Lo_2_Carbon',
        'Ton_Kho_Than', 'Ton_Kho_Chat_Dot', 'Ton_Kho_Carbon'
    ]

    for col in cols_to_fill:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(',', '', regex=False),
                    errors='coerce'
                )
                .fillna(0)
            )
        else:
            df[col] = 0

    df['Total_Than'] = df['Lo_1_Than'] + df['Lo_2_Than']
    df['Total_Chat_Dot'] = df['Lo_1_Chat_Dot'] + df['Lo_2_Chat_Dot']
    df['Total_Carbon'] = df['Lo_1_Carbon'] + df['Lo_2_Carbon']

    return df


def feature_engineering(df, n_lags=3):
    """
    BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING)
    Tạo độ trễ (Lags) và Trung bình trượt (Rolling Means) để bắt xu hướng.
    """
    print("⏳ [2/5] Đang kiến tạo đặc trưng (Feature Engineering)...")
    data = df.copy()
    data['DayOfWeek'] = data['Ngay'].dt.dayofweek

    for i in range(1, n_lags + 1):
        data[f'Than_lag_{i}'] = data['Total_Than'].shift(i)
        data[f'Chat_Dot_lag_{i}'] = data['Total_Chat_Dot'].shift(i)
        data[f'Carbon_lag_{i}'] = data['Total_Carbon'].shift(i)

    data['Than_RollMean_3'] = data['Total_Than'].shift(1).rolling(window=3).mean()
    data['ChatDot_RollMean_3'] = data['Total_Chat_Dot'].shift(1).rolling(window=3).mean()
    data['Carbon_RollMean_3'] = data['Total_Carbon'].shift(1).rolling(window=3).mean()

    return data.dropna().reset_index(drop=True)


def train_and_tune_model(data, n_lags=3):
    """
    BƯỚC 3 & 4: HUẤN LUYỆN VÀ TỐI ƯU HÓA SIÊU THAM SỐ
    Ứng dụng GridSearch và Time-Series Cross Validation.
    """
    print("⏳ [3/5] Đang chia tập dữ liệu (Train/Test Split) theo không gian thời gian...")

    feature_cols = (
        ['DayOfWeek', 'Than_RollMean_3', 'ChatDot_RollMean_3', 'Carbon_RollMean_3']
        + [f'Than_lag_{i}' for i in range(1, n_lags + 1)]
        + [f'Chat_Dot_lag_{i}' for i in range(1, n_lags + 1)]
        + [f'Carbon_lag_{i}' for i in range(1, n_lags + 1)]
    )
    target_cols = ['Total_Than', 'Total_Chat_Dot', 'Total_Carbon']

    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    X_train, y_train = train_data[feature_cols], train_data[target_cols]
    X_test, y_test = test_data[feature_cols], test_data[target_cols]

    print("⏳ [4/5] Đang dò tìm Siêu tham số tối ưu (Hyperparameter Tuning) bằng GridSearchCV...")

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"   -> Đã tìm ra cấu hình tốt nhất: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    print("\n" + "=" * 50)
    print(" BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (TRÊN TẬP KIỂM THỬ OOT)")
    print("=" * 50)
    print(f"{'Vật Tư':<15} | {'MAE (Tấn)':<12} | {'RMSE (Tấn)':<12}")
    print("-" * 50)
    print(f"{'Than':<15} | {mae[0]:<12.2f} | {rmse[0]:<12.2f}")
    print(f"{'Chất Đốt':<15} | {mae[1]:<12.2f} | {rmse[1]:<12.2f}")
    print(f"{'Carbon':<15} | {mae[2]:<12.2f} | {rmse[2]:<12.2f}")
    print("=" * 50 + "\n")

    return best_model, data, target_cols, feature_cols


def _normalize_forecast_dates(forecast_dates):
    normalized = pd.to_datetime(list(forecast_dates)).normalize()
    if len(normalized) != 7:
        raise ValueError("forecast_dates phải chứa đúng 7 ngày.")

    if normalized.duplicated().any():
        raise ValueError("forecast_dates không được chứa ngày trùng lặp.")

    expected = pd.date_range(normalized[0], periods=7, freq='D')
    if not normalized.equals(expected):
        raise ValueError("forecast_dates phải là 7 ngày liên tiếp theo thứ tự tăng dần.")

    return normalized


def forecast_next_7_days(
    model,
    data,
    target_cols,
    feature_cols,
    n_lags=3,
    forecast_dates=None
):
    """
    BƯỚC 5: DỰ BÁO CUỐN CHIẾU (ROLLING FORECAST).

    Khi forecast_dates được truyền vào, mô hình dùng chính xác 7 ngày đó.
    Điều này cho phép main_pipeline đồng bộ Demand Forecast với Weather Forecast.
    Nếu không truyền, giữ hành vi cũ: dự báo 7 ngày ngay sau ngày dữ liệu cuối.
    """
    print("⏳ [5/5] Đang chạy mô phỏng dự báo cuốn chiếu cho 7 ngày tới...")

    if forecast_dates is None:
        last_date = pd.Timestamp(data['Ngay'].iloc[-1]).normalize()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq='D')
    else:
        forecast_dates = _normalize_forecast_dates(forecast_dates)

    buffer_than = data['Total_Than'].tail(n_lags).values.tolist()
    buffer_cd = data['Total_Chat_Dot'].tail(n_lags).values.tolist()
    buffer_cb = data['Total_Carbon'].tail(n_lags).values.tolist()

    ton_kho_than = float(data['Ton_Kho_Than'].iloc[-1])
    ton_kho_chat_dot = float(data['Ton_Kho_Chat_Dot'].iloc[-1])
    ton_kho_carbon = float(data['Ton_Kho_Carbon'].iloc[-1])

    forecast_results = []

    for future_date in forecast_dates:
        rm_than = np.mean(buffer_than[-3:])
        rm_cd = np.mean(buffer_cd[-3:])
        rm_cb = np.mean(buffer_cb[-3:])

        x_input_dict = {
            'DayOfWeek': [future_date.dayofweek],
            'Than_RollMean_3': [rm_than],
            'ChatDot_RollMean_3': [rm_cd],
            'Carbon_RollMean_3': [rm_cb],
            'Than_lag_1': [buffer_than[-1]],
            'Than_lag_2': [buffer_than[-2]],
            'Than_lag_3': [buffer_than[-3]],
            'Chat_Dot_lag_1': [buffer_cd[-1]],
            'Chat_Dot_lag_2': [buffer_cd[-2]],
            'Chat_Dot_lag_3': [buffer_cd[-3]],
            'Carbon_lag_1': [buffer_cb[-1]],
            'Carbon_lag_2': [buffer_cb[-2]],
            'Carbon_lag_3': [buffer_cb[-3]]
        }

        x_input = pd.DataFrame(x_input_dict)[feature_cols]
        preds = model.predict(x_input)[0]

        pred_than = max(0.0, float(preds[0]))
        pred_cd = max(0.0, float(preds[1]))
        pred_cb = max(0.0, float(preds[2]))

        ton_kho_than -= pred_than
        ton_kho_chat_dot -= pred_cd
        ton_kho_carbon -= pred_cb

        forecast_results.append({
            'Ngay': future_date.strftime('%Y-%m-%d'),
            'Dùng_Than': round(pred_than, 1),
            'TỒN_THAN': round(ton_kho_than, 1),
            'Dùng_C.Đốt': round(pred_cd, 1),
            'TỒN_C.ĐỐT': round(ton_kho_chat_dot, 1),
            'Dùng_Carbon': round(pred_cb, 1),
            'TỒN_CARBON': round(ton_kho_carbon, 1)
        })

        buffer_than.append(pred_than)
        buffer_cd.append(pred_cd)
        buffer_cb.append(pred_cb)

    return pd.DataFrame(forecast_results)


def run_demand_forecast(
    data_path="Data.csv",
    output_path="D_t_forecast_7days.csv",
    forecast_dates=None
):
    """
    Chạy toàn bộ Lớp 1 và lưu demand forecast.
    main_pipeline truyền canonical 7-day horizon qua forecast_dates.
    """
    df_raw = load_and_preprocess(data_path)
    if df_raw is None:
        raise FileNotFoundError(data_path)

    df_features = feature_engineering(df_raw)
    best_model, df_ready, targets, features = train_and_tune_model(df_features)

    df_forecast = forecast_next_7_days(
        best_model,
        df_ready,
        targets,
        features,
        forecast_dates=forecast_dates
    )
    df_forecast.to_csv(output_path, index=False)

    print("=== KẾT QUẢ DỰ BÁO D_t & TỒN KHO DỰ KIẾN (7 NGÀY TỚI) ===")
    print(df_forecast.to_string(index=False))
    print("==========================================================")
    print(f"✅ Hoàn thành quy trình ML! Đã lưu: '{output_path}'")

    return df_forecast


if __name__ == "__main__":
    print("\n" + "★" * 60)
    print(" HỆ THỐNG DỰ BÁO CHUỖI THỜI GIAN (THESIS GRADE PIPELINE)")
    print("★" * 60 + "\n")
    run_demand_forecast()
