import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

def build_fuzzy_system_thesis(visualize=True):
    """
    Xây dựng Hệ thống Logic Mờ chuẩn học thuật với các Hàm liên thuộc (MFs) tự định nghĩa.
    """
    # 1. KHÔNG GIAN NỀN (UNIVERSE OF DISCOURSE)
    demand = ctrl.Antecedent(np.arange(0, 101, 1), 'Demand') # % Công suất
    risk = ctrl.Antecedent(np.arange(0, 11, 1), 'Risk')      # Điểm rủi ro
    safety_days = ctrl.Consequent(np.arange(1, 8, 0.1), 'Safety_Days', defuzzify_method='centroid')

    # 2. ĐỊNH NGHĨA HÀM LIÊN THUỘC (THESIS GRADE)
    # Hàm Demand: Hình thang (trapmf) cho Thấp/Cao, Hình tam giác (trimf) cho TB
    demand['Thap'] = fuzz.trapmf(demand.universe, [0, 0, 30, 50])
    demand['TB'] = fuzz.trimf(demand.universe, [30, 50, 70])
    demand['Cao'] = fuzz.trapmf(demand.universe, [50, 70, 100, 100])

    # Hàm Risk
    risk['Thap'] = fuzz.trapmf(risk.universe, [0, 0, 3, 5])
    risk['TB'] = fuzz.trimf(risk.universe, [3, 5, 7])
    risk['Cao'] = fuzz.trapmf(risk.universe, [5, 7, 10, 10])

    # Hàm Safety Days (Ngày an toàn)
    safety_days['Ngan'] = fuzz.trapmf(safety_days.universe, [1, 1, 2, 4])
    safety_days['Vua'] = fuzz.trimf(safety_days.universe, [2, 4, 6])
    safety_days['Dai'] = fuzz.trapmf(safety_days.universe, [4, 6, 7, 7])

    # 3. LUẬT MỜ (FUZZY RULES) - MA TRẬN ĐẦY ĐỦ (3x3 = 9 Luật)
    # Vét cạn mọi trường hợp để hệ thống không có "điểm mù" (No logical holes)
    
    # Nhóm 1: Khi Nhu cầu THẤP
    rule1 = ctrl.Rule(demand['Thap'] & risk['Thap'], safety_days['Ngan'])
    rule2 = ctrl.Rule(demand['Thap'] & risk['TB'], safety_days['Ngan'])
    rule3 = ctrl.Rule(demand['Thap'] & risk['Cao'], safety_days['Dai'])
    
    # Nhóm 2: Khi Nhu cầu TRUNG BÌNH (TB)
    rule4 = ctrl.Rule(demand['TB'] & risk['Thap'], safety_days['Ngan'])
    rule5 = ctrl.Rule(demand['TB'] & risk['TB'], safety_days['Vua'])
    rule6 = ctrl.Rule(demand['TB'] & risk['Cao'], safety_days['Dai'])
    
    # Nhóm 3: Khi Nhu cầu CAO
    rule7 = ctrl.Rule(demand['Cao'] & risk['Thap'], safety_days['Vua'])
    rule8 = ctrl.Rule(demand['Cao'] & risk['TB'], safety_days['Dai'])
    rule9 = ctrl.Rule(demand['Cao'] & risk['Cao'], safety_days['Dai'])

    # 4. TẠO BỘ ĐIỀU KHIỂN (Nạp đủ 9 luật vào)
    fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    
    # ==========================================
    # 4. KẾT XUẤT ĐỒ THỊ CHUẨN LUẬN VĂN (3D SURFACE)
    # ==========================================
    if visualize:
        print("🎨 Đang vẽ Không gian điều khiển mờ (3D Control Surface)...")
        # Tạo lưới dữ liệu mô phỏng
        sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
        x_demand = np.arange(0, 101, 5)
        y_risk = np.arange(0, 11, 0.5)
        X, Y = np.meshgrid(x_demand, y_risk)
        Z = np.zeros_like(X)
        
        # Tính toán bề mặt
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                sim.input['Demand'] = X[i, j]
                sim.input['Risk'] = Y[i, j]
                sim.compute()
                Z[i, j] = sim.output['Safety_Days']
                
        # Vẽ 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Nhu cầu (% Công suất)', fontweight='bold')
        ax.set_ylabel('Rủi ro (0-10)', fontweight='bold')
        ax.set_zlabel('Ngày dự trữ an toàn', fontweight='bold')
        ax.set_title('BỀ MẶT ĐIỀU KHIỂN LOGIC MỜ\n(Fuzzy Logic Control Surface)', fontweight='bold')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.savefig("Fuzzy_3D_Surface.png", dpi=300, bbox_inches='tight')
        print("✅ Đã xuất: 'Fuzzy_3D_Surface.png' (Dùng chèn Luận văn)")
        plt.close()

    return ctrl.ControlSystemSimulation(fuzzy_ctrl)

def calculate_dynamic_safety_stock(forecast_file, risk_score=5):
    """
    Tính toán I_safe,t (Tồn kho an toàn) dựa trên Logic mờ trọng tâm.
    """
    print("\n🧠 KHỞI ĐỘNG LỚP 2: FUZZY LOGIC CONTROLLER (THESIS GRADE)")
    
    # Khởi tạo hệ thống và tự động xuất ảnh 3D
    fuzzy_sim = build_fuzzy_system_thesis(visualize=True)
    
    try:
        df = pd.read_csv(forecast_file)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy '{forecast_file}'.")
        return
    
    # Định mức công suất
    MAX_CAPACITY = {'Than': 1000.0, 'ChatDot': 500.0, 'Carbon': 200.0}
    safe_stocks = []
    
    print(f"🌦️ Đang áp dụng Rủi ro vận tải: {risk_score}/10")
    print("⚙️ Giải mờ (Centroid Defuzzification) cho 7 ngày tới...\n")
    
    for index, row in df.iterrows():
        dt_than, dt_cd, dt_cb = row.get('Dùng_Than', 0), row.get('Dùng_C.Đốt', 0), row.get('Dùng_Carbon', 0)
        
        def get_safety_stock(dt_value, max_cap):
            if max_cap == 0 or dt_value == 0: return 0, 0
            demand_pct = min(100, max(0, (dt_value / max_cap) * 100))
            fuzzy_sim.input['Demand'] = demand_pct
            fuzzy_sim.input['Risk'] = risk_score
            fuzzy_sim.compute()
            s_days = fuzzy_sim.output['Safety_Days']
            return s_days, dt_value * s_days
        
        days_than, isafe_than = get_safety_stock(dt_than, MAX_CAPACITY['Than'])
        days_cd, isafe_cd = get_safety_stock(dt_cd, MAX_CAPACITY['ChatDot'])
        days_cb, isafe_cb = get_safety_stock(dt_cb, MAX_CAPACITY['Carbon'])
        
        safe_stocks.append({
            'Ngay': row['Ngay'],
            'I_safe_Than': round(isafe_than, 1),
            'I_safe_CĐốt': round(isafe_cd, 1),
            'I_safe_Carbon': round(isafe_cb, 1)
        })
        
    df_isafe = pd.DataFrame(safe_stocks)
    print("=== TỒN KHO AN TOÀN ĐỘNG (I_safe,t) CHO 7 NGÀY TỚI ===")
    print(df_isafe.to_string(index=False))
    print("======================================================")
    
    df_isafe.to_csv("I_safe_t_dynamic.csv", index=False)
    print("✅ Đã lưu kết quả thành công!")

if __name__ == "__main__":
    calculate_dynamic_safety_stock("D_t_forecast_7days.csv", risk_score=8.0)