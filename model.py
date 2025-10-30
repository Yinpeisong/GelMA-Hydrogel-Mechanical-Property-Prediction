import torch
import joblib
import numpy as np

# 你的模型定义
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class BNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(3, 32)
        self.fc2 = BayesianLinear(32, 16)
        self.out = BayesianLinear(16, 2)  # 输出两个特征
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(x)

def load_model(model_path, device="cpu"):
    model = BNNModel().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    return model

def preprocess_input(X):
    X = (X - np.array([5.0, 0.5, 30.0])) / np.array([2.0, 0.2, 15.0])  # 示例标准化
    return torch.tensor(X, dtype=torch.float32)
