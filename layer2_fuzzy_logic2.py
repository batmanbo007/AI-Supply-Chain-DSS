import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

def plot_thesis_fuzzy_charts():
    print("🎨 Đang kết xuất các biểu đồ Học thuật cho Lớp 2...")

    # 1. TÁI TẠO KHÔNG GIAN MỜ
    x_demand = np.arange(0, 101, 1)
    x_risk = np.arange(0, 11, 1)
    x_safety = np.arange(1, 8, 0.1)

    # Khai báo Hàm liên thuộc
    dem_thap = fuzz.trapmf(x_demand, [0, 0, 30, 50])
    dem_tb = fuzz.trimf(x_demand, [30, 50, 70])
    dem_cao = fuzz.trapmf(x_demand, [50, 70, 100, 100])

    risk_thap = fuzz.trapmf(x_risk, [0, 0, 3, 5])
    risk_tb = fuzz.trimf(x_risk, [3, 5, 7])
    risk_cao = fuzz.trapmf(x_risk, [5, 7, 10, 10])

    safe_ngan = fuzz.trapmf(x_safety, [1, 1, 2, 4])
    safe_vua = fuzz.trimf(x_safety, [2, 4, 6])
    safe_dai = fuzz.trapmf(x_safety, [4, 6, 7, 7])

    # ==========================================
    # BIỂU ĐỒ 1: HÀM LIÊN THUỘC (MEMBERSHIP FUNCTIONS)
    # ==========================================
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(10, 9))
    fig.canvas.manager.set_window_title('Đồ thị Hàm liên thuộc (Membership Functions)')

    # Biến Nhu cầu
    ax0.plot(x_demand, dem_thap, 'b', linewidth=2, label='Thấp')
    ax0.plot(x_demand, dem_tb, 'g', linewidth=2, label='Trung Bình')
    ax0.plot(x_demand, dem_cao, 'r', linewidth=2, label='Cao')
    ax0.set_title('Biến Đầu Vào 1: Nhu Cầu Tiêu Thụ (% Công suất)', fontweight='bold')
    ax0.legend()
    ax0.grid(True, linestyle='--', alpha=0.6)

    # Biến Rủi ro
    ax1.plot(x_risk, risk_thap, 'b', linewidth=2, label='Thấp')
    ax1.plot(x_risk, risk_tb, 'g', linewidth=2, label='Trung Bình')
    ax1.plot(x_risk, risk_cao, 'r', linewidth=2, label='Cao')
    ax1.set_title('Biến Đầu Vào 2: Rủi Ro Vận Tải (Thang 0-10)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Biến Đầu ra
    ax2.plot(x_safety, safe_ngan, 'b', linewidth=2, label='Ngắn')
    ax2.plot(x_safety, safe_vua, 'g', linewidth=2, label='Vừa')
    ax2.plot(x_safety, safe_dai, 'r', linewidth=2, label='Dài')
    ax2.set_title('Biến Đầu Ra: Số Ngày Dự Trữ An Toàn Bắt Buộc (Safety Days)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("Fuzzy_Membership_Functions.png", dpi=300)
    
    # Hiển thị biểu đồ 2D lên màn hình
    plt.show()

    # ==========================================
    # BIỂU ĐỒ 2: MẶT PHẲNG ĐIỀU KHIỂN 3D (Tương tác được)
    # ==========================================
    print("⏳ Tiếp tục kết xuất Đồ thị 3D Surface...")
    
    # Khai báo lại hệ thống bằng thư viện skfuzzy để mô phỏng
    demand = ctrl.Antecedent(x_demand, 'Demand')
    risk = ctrl.Antecedent(x_risk, 'Risk')
    safety_days = ctrl.Consequent(x_safety, 'Safety_Days', defuzzify_method='centroid')

    demand['Thap'], demand['TB'], demand['Cao'] = dem_thap, dem_tb, dem_cao
    risk['Thap'], risk['TB'], risk['Cao'] = risk_thap, risk_tb, risk_cao
    safety_days['Ngan'], safety_days['Vua'], safety_days['Dai'] = safe_ngan, safe_vua, safe_dai

    rule1 = ctrl.Rule(demand['Thap'] & risk['Thap'], safety_days['Ngan'])
    rule2 = ctrl.Rule(demand['Thap'] & risk['TB'], safety_days['Ngan'])
    rule3 = ctrl.Rule(demand['Thap'] & risk['Cao'], safety_days['Dai'])
    rule4 = ctrl.Rule(demand['TB'] & risk['Thap'], safety_days['Ngan'])
    rule5 = ctrl.Rule(demand['TB'] & risk['TB'], safety_days['Vua'])
    rule6 = ctrl.Rule(demand['TB'] & risk['Cao'], safety_days['Dai'])
    rule7 = ctrl.Rule(demand['Cao'] & risk['Thap'], safety_days['Vua'])
    rule8 = ctrl.Rule(demand['Cao'] & risk['TB'], safety_days['Dai'])
    rule9 = ctrl.Rule(demand['Cao'] & risk['Cao'], safety_days['Dai'])

    fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

    X, Y = np.meshgrid(np.arange(0, 101, 5), np.arange(0, 11, 0.5))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            sim.input['Demand'] = X[i, j]
            sim.input['Risk'] = Y[i, j]
            sim.compute()
            Z[i, j] = sim.output['Safety_Days']
            
    fig = plt.figure(figsize=(10, 7))
    fig.canvas.manager.set_window_title('Mặt phẳng không gian mờ 3D')
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('\nNhu cầu (% Công suất)', fontweight='bold')
    ax.set_ylabel('\nRủi ro (0-10)', fontweight='bold')
    ax.set_zlabel('\nNgày dự trữ an toàn', fontweight='bold')
    ax.set_title('BỀ MẶT ĐIỀU KHIỂN LOGIC MỜ 3D', fontweight='bold')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Hiện biểu đồ 3D lên màn hình (bạn có thể dùng chuột xoay các góc nhìn)
    plt.show()

if __name__ == "__main__":
    plot_thesis_fuzzy_charts()