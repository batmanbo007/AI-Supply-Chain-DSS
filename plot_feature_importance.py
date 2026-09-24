import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Tái sử dụng các hàm đã chuẩn hóa từ Layer_1
from layer1_machinelearning import load_and_preprocess, feature_engineering

def plot_feature_importance():
    print("📊 Đang khởi tạo mô hình và tính toán trọng số Đặc trưng...")

    # 1. Chuẩn bị dữ liệu (Dùng chính logic của Layer 1)
    df_raw = load_and_preprocess("Data.csv")
    if df_raw is None:
        return
    
    data = feature_engineering(df_raw)
    
    # 2. Khai báo biến
    n_lags = 3
    feature_cols = ['DayOfWeek', 'Than_RollMean_3', 'ChatDot_RollMean_3', 'Carbon_RollMean_3'] + \
                   [f'Than_lag_{i}' for i in range(1, n_lags + 1)] + \
                   [f'Chat_Dot_lag_{i}' for i in range(1, n_lags + 1)] + \
                   [f'Carbon_lag_{i}' for i in range(1, n_lags + 1)]
    target_cols = ['Total_Than', 'Total_Chat_Dot', 'Total_Carbon']
    
    X = data[feature_cols]
    y = data[target_cols]
    
    # 3. Huấn luyện mô hình để lấy trọng số (Gini Importance)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # 4. Trích xuất độ quan trọng
    importances = model.feature_importances_
    
    # Đóng gói vào DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    
    # Sắp xếp từ cao xuống thấp
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    
    # ==========================================
    # 5. VẼ BIỂU ĐỒ CHUẨN HỌC THUẬT (ACADEMIC STYLE)
    # ==========================================
    print("🎨 Đang kết xuất (Render) biểu đồ...")
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid") # Hình nền lưới trắng chuyên nghiệp
    
    # Vẽ Barplot với dải màu Viridis chuẩn khoa học
    ax = sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    
    # Căn chỉnh nhãn mác
    plt.title('TẦM QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG ĐẦU VÀO (FEATURE IMPORTANCE)\nRandom Forest Multi-Output Regressor', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Mức độ đóng góp (Gini Importance)', fontsize=12, fontweight='bold')
    plt.ylabel('Các biến đầu vào (Features)', fontsize=12, fontweight='bold')
    
    # Thêm số liệu trực tiếp lên đuôi mỗi cột
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.002, p.get_y() + p.get_height()/2. + 0.1, 
                 '{:1.3f}'.format(width), ha="left", fontsize=10, color='black')
        
    plt.xlim(0, max(importances) + 0.05) # Mở rộng lề phải để không bị lẹm số
    plt.tight_layout()
    
    # 6. Xuất ra file ảnh chất lượng cao (300 DPI) để chèn Word/LaTeX
    filename = "Feature_Importance_Chart.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"✅ Đã xuất biểu đồ thành công! File lưu tại: '{filename}'")
    
    # Hiển thị biểu đồ lên màn hình trực tiếp
    plt.show()

if __name__ == "__main__":
    plot_feature_importance()