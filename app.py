import streamlit as st
import torch
import numpy as np
from model.model_utils import load_model, preprocess_input

# 页面配置
st.set_page_config(
    page_title="水凝胶预测系统",
    page_icon="🧫",
    layout="centered"
)

# 全局样式美化
st.markdown("""
    <style>
    .big-font {font-size: 24px; font-weight: bold; color: #333;}
    .card {
        background-color: #f8f9fc;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    .metric-label {color: #555; font-size: 16px;}
    .metric-value {font-size: 22px; font-weight: bold; color: #3751ff;}
    .section-title {font-size: 20px; font-weight: bold; color: #333;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">水凝胶 BNN 预测系统</p>', unsafe_allow_html=True)
st.write("输入原料参数，模拟 BNN 模型预测水凝胶性能和生物相容性。")

# 加载模型
@st.cache_resource
def get_model():
    model = load_model("ModulusOutput/bnn_trained.pth", device="cpu")
    model.eval()
    return model

model = get_model()

# -----------------------------
# 输入参数区域
# -----------------------------
st.markdown('<div class="card"><p class="section-title">原料参数输入 (Inputs)</p>', unsafe_allow_html=True)

gelma = st.slider("水凝胶溶液浓度 (%)", 1.0, 10.0, 4.7, step=0.1)
lap = st.slider("LAP 浓度 (%)", 0.01, 1.0, 0.5, step=0.01)
uv_time = st.slider("光交联时间 (秒)", 1, 60, 30, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 模型预测
# -----------------------------
if st.button("运行 BNN 预测"):
    X_input = np.array([[gelma, lap, uv_time]])
    X_tensor = preprocess_input(X_input)

    # Monte Carlo Sampling
    preds = []
    with torch.no_grad():
        for _ in range(50):
            y_pred = model(X_tensor)
            preds.append(y_pred.cpu().numpy())
    preds = np.vstack(preds)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    # 示例假设输出有两个：模量 & 应力
    hydro_modulus = mean_pred[0]
    yield_stress = mean_pred[1]
    hydro_std = std_pred[0]
    stress_std = std_pred[1]

    # 细胞铺展状态推断（示例规则）
    if hydro_modulus < 10:
        cell_status = "低刚度 - 限制铺展"
    elif 10 <= hydro_modulus <= 25:
        cell_status = "适中刚度 - 有利铺展"
    else:
        cell_status = "高刚度 - 抑制生长"

    # -----------------------------
    # 输出结果区域
    # -----------------------------
    st.markdown('<div class="card"><p class="section-title">预测结果 (Outputs)</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="metric-label">水凝胶模量 (kPa)</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{hydro_modulus:.2f} ± {hydro_std:.2f}</p>', unsafe_allow_html=True)
        st.caption("BNN预测均值 ± 标准差")

    with col2:
        st.markdown('<p class="metric-label">临界应力 (kPa)</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{yield_stress:.2f} ± {stress_std:.2f}</p>', unsafe_allow_html=True)
        st.caption("BNN预测均值 ± 标准差")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f"**细胞铺展状态：** {cell_status}")

    st.markdown('</div>', unsafe_allow_html=True)
