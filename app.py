import streamlit as st
import torch
import numpy as np
import joblib

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

# ======================
# 模型定义
# ======================
@variational_estimator
class BNNRegressionModel(torch.nn.Module):
    """预测储能模量(G')和损耗模量(G'')"""
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(3, 32)
        self.fc2 = BayesianLinear(32, 16)
        self.out = BayesianLinear(16, 2)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)


@variational_estimator
class BNNClassificationModel(torch.nn.Module):
    """预测临界应力（分类）"""
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(3, 32)
        self.fc2 = BayesianLinear(32, 16)
        self.out = BayesianLinear(16, 3)  # 三类：0~10Pa、10~50Pa、>50Pa
        self.act = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.softmax(self.out(x))


# ======================
# 工具函数
# ======================
def load_model(model_path, model_class, device="cpu"):
    model = model_class().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def preprocess_input(X):
    """标准化输入（与训练一致）"""
    X = (X - np.array([5.0, 0.5, 30.0])) / np.array([2.0, 0.2, 15.0])
    return torch.tensor(X, dtype=torch.float32)

def predict_with_uncertainty(model, X, n_samples=30):
    """多次前向传播获取平均预测与不确定性"""
    preds = [model(X).detach().numpy() for _ in range(n_samples)]
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std


# ======================
# Streamlit 界面
# ======================
st.set_page_config(page_title="Hydrogel Mechanics Predictor", page_icon="🧫", layout="centered")

st.title("🧫 水凝胶力学性能与细胞生长预测平台")
st.markdown("""
该应用通过贝叶斯神经网络（BNN）模型预测 **GelMA水凝胶的力学性能**，包括：
- 储能模量 (G′)
- 损耗模量 (G″)
- 临界应力 (Critical Stress)
并根据力学特性综合预测**细胞生长情况**。
""")

st.divider()

# 输入区域
st.header("🔧 输入水凝胶配方参数")
col1, col2, col3 = st.columns(3)
gelma = col1.number_input("GelMA浓度 (%)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
lap = col2.number_input("LAP浓度 (%)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
uv = col3.number_input("光交联时间 (s)", min_value=5.0, max_value=120.0, value=30.0, step=5.0)

# 模型路径
reg_model_path = "bnn_trained.pth"
cls_model_path = "bnn_trained.pth"

device = "cpu"

# 加载模型
try:
    reg_model = load_model(reg_model_path, BNNRegressionModel, device)
    cls_model = load_model(cls_model_path, BNNClassificationModel, device)
except Exception as e:
    st.error("❌ 模型加载失败，请检查路径或文件格式！")
    st.stop()

# 预测按钮
if st.button("🚀 开始预测"):
    with st.spinner("模型预测中，请稍候..."):
        X = np.array([[gelma, lap, uv]])
        X_tensor = preprocess_input(X).to(device)

        # 模量预测
        reg_mean, reg_std = predict_with_uncertainty(reg_model, X_tensor)
        Gp, Gpp = reg_mean[0]
        Gp_std, Gpp_std = reg_std[0]

        # 临界应力预测
        cls_mean, _ = predict_with_uncertainty(cls_model, X_tensor)
        critical_class = int(np.argmax(cls_mean))
        stress_classes = ["0–10 Pa", "10–50 Pa", ">50 Pa"]
        critical_label = stress_classes[critical_class]

        # 细胞生长预测逻辑
        if Gp < 500 and critical_class == 0:
            growth = "🌱 模量较低且临界应力适中，有利于细胞生长"
        elif Gp < 1000 and critical_class == 1:
            growth = "🧬 力学性质中等，可支持部分细胞黏附与增殖"
        else:
            growth = "⚠️ 模量较高或临界应力过大，不利于细胞扩散"

    st.success("✅ 预测完成！")
    st.divider()

    # 输出结果
    st.subheader("📊 预测结果")
    st.write(f"**储能模量 G′:** {Gp:.2f} ± {Gp_std:.2f} Pa")
    st.write(f"**损耗模量 G″:** {Gpp:.2f} ± {Gpp_std:.2f} Pa")
    st.write(f"**临界应力区间:** {critical_label}")
    st.info(growth)

    # 可视化
    st.subheader("📈 模型不确定性可视化")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(["G′", "G″"], [Gp, Gpp], yerr=[Gp_std, Gpp_std], capsize=5)
    ax.set_ylabel("Modulus (Pa)")
    ax.set_title("储能与损耗模量预测 (含不确定性)")
    st.pyplot(fig)

st.divider()
st.caption("Developed with ❤️ using Bayesian Neural Networks (BNN) & Streamlit.")
