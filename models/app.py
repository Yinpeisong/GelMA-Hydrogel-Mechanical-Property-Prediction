import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import joblib  # 用于加载 scaler

# ======================
# 1️⃣ 定义模型结构
# ======================
@variational_estimator
class BNNRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(in_features=3, out_features=64)
        self.fc2 = BayesianLinear(in_features=64, out_features=32)
        self.fc3 = BayesianLinear(in_features=32, out_features=16)
        self.out = BayesianLinear(in_features=16, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

@variational_estimator
class BNNClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=32)
        self.fc2 = torch.nn.Linear(in_features=32, out_features=16)
        self.fc3 = torch.nn.Linear(in_features=16, out_features=3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ======================
# 2️⃣ 工具函数
# ======================
def load_model(model_path, model_class, device="cpu"):
    """加载模型"""
    model = model_class().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def predict_with_uncertainty_regression(model, X_tensor, scaler_y, n_samples=30):
    """回归模型 Monte Carlo 预测 + 反标准化"""
    preds_all = []
    with torch.no_grad():
        for _ in range(n_samples):
            y_sample = model(X_tensor).detach().cpu().numpy()
            y_real = scaler_y.inverse_transform(y_sample)
            preds_all.append(y_real)
    preds_all = np.array(preds_all)
    mean = preds_all.mean(axis=0)[0]
    std = preds_all.std(axis=0)[0]
    return mean, std

# ======================
# 3️⃣ Streamlit 页面设置
# ======================
st.set_page_config(page_title="Hydrogel Mechanics Predictor", page_icon="🧫", layout="centered")
st.title("🧫 Hydrogel Mechanics & Cell Growth Predictor")

st.markdown("""
This application is based on a **Bayesian Neural Network (BNN)** model to predict the 
mechanical properties and cell growth performance of GelMA hydrogels, including:

- Storage Modulus (G′)
- Loss Modulus (G″)
- Critical Stress
- Cell Growth Adaptability Prediction
""")
st.divider()

# ======================
# 4️⃣ 输入区域
# ======================
st.header("🔧 Input hydrogel formulation parameters (please convert percentages to decimals)")
col1, col2, col3 = st.columns(3)
gelma = col1.number_input("GelMA Concentration", min_value=0.01, max_value=0.2, value=0.03, step=0.005, format="%.4f")
lap = col2.number_input("LAP Concentration", min_value=0.0001, max_value=0.1, value=0.0002, step=0.0001, format="%.4f")
uv = col3.number_input("UV Duration (s)", min_value=5.0, max_value=1000.0, value=30.0, step=1.0, format="%.2f")

# ======================
# 5️⃣ 加载模型与scaler
# ======================
device = "cpu"
reg_model_path = "models/bnn_model_regression.pth"
cls_model_path = "models/bnn_model_classifier.pth"
scaler_X_path = "models/regression_scaler_X.pkl"
scaler_y_path = "models/regression_scaler_y.pkl"

try:
    reg_model = load_model(reg_model_path, BNNRegressionModel, device)
    cls_model = load_model(cls_model_path, BNNClassificationModel, device)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model or Scaler loading failed. Please check the file path or format.\n\nError：{e}")
    st.stop()

# ======================
# 6️⃣ 模型预测逻辑
# ======================
if st.button("🚀 Start Prediction"):
    with st.spinner("正在进行贝叶斯推断，请稍候..."):
        # ----------------------
        # 回归输入标准化
        # ----------------------
        X_input = np.array([[gelma, lap, uv]])
        X_scaled = scaler_X.transform(X_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        # ----------------------
        # 回归预测 (G′, G″)
        # ----------------------
        Gp_Gpp_mean, Gp_Gpp_std = predict_with_uncertainty_regression(reg_model, X_tensor, scaler_y)
        Gp, Gpp = Gp_Gpp_mean
        Gp_std, Gpp_std = Gp_Gpp_std if isinstance(Gp_Gpp_std, (list, np.ndarray)) else (Gp_Gpp_std, Gp_Gpp_std)

        # ----------------------
        # 分类预测 (Critical Stress)
        # ----------------------
        cls_output = cls_model(X_tensor).detach().cpu().numpy()
        critical_class = int(np.argmax(cls_output))
        stress_classes = ["0–10 Pa", "10–30 Pa", ">30 Pa"]
        critical_label = stress_classes[critical_class]

        # ----------------------
        # 综合细胞生长判断
        # ----------------------
        if Gp < 100 and critical_class == 0:
            growth = "🌱 Low modulus and moderate critical stress — favorable for cell spreading and proliferation"
        elif Gp < 300 and critical_class == 1:
            growth = "🧬 Moderate mechanical properties — supportive of cell adhesion and partial proliferation"
        else:
            growth = "⚠️ High modulus or excessive critical stress — unfavorable for cell spreading and growth"

    # ======================
    # 7️⃣ 输出结果
    # ======================
    st.success("✅ Prediction Completed!")
    st.divider()
    st.subheader("📊 Model Prediction Results")
    st.write(f"**Storage Modulus G′:** {Gp:.2f} ± {Gp_std:.2f} Pa")
    st.write(f"**Loss Modulus G″:** {Gpp:.2f} ± {Gpp_std:.2f} Pa")
    st.write(f"**Critical Stress Range:** {critical_label}")
    st.info(growth)

    # ======================
    # 8️⃣ 可视化部分
    # ======================
    st.subheader("📈 Model Prediction Visualization (with Uncertainty)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["G′", "G″"], [Gp, Gpp], yerr=[Gp_std, Gpp_std],
           color=["#66b3ff", "#99ff99"], capsize=5)
    ax.set_ylabel("Modulus (Pa)", fontweight="bold")
    ax.set_title("Modulus Prediction (Uncertainty)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

st.divider()
st.caption("Developed with ❤️ using Bayesian Neural Networks (BNN) & Streamlit.")
