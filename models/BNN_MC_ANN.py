import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from blitz.modules import BayesianLinear
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rc('font', family='Arial')   #设置全局字体为Arial
plt.rcParams.update({'font.size': 18,  'font.weight': 'bold'})     #设置全局字体大小为16

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 读取数据
data = pd.read_csv('dataset/1010modulus.csv')
X_raw = data[['GM浓度', 'LAP浓度', '光照时间']].values
y_raw = data[['储能模量', '损耗模量']].values

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw)

# 转 tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 贝叶斯神经网络
class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(in_features=input_dim, out_features=64)
        self.fc2 = BayesianLinear(in_features=64, out_features=32)
        self.fc3 = BayesianLinear(in_features=32, out_features=16)
        self.dropout = nn.Dropout(p=0.1)
        self.out = BayesianLinear(in_features=16, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.out(x)

def train_bnn(model, X_train, y_train, epochs=1000, lr=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return loss_history

bnn_model = BayesianNN(input_dim=3, output_dim=2)
bnn_loss = train_bnn(bnn_model, X_train, y_train)


# BNN Loss
plt.figure(figsize=(8, 5))
plt.plot(bnn_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BNN Loss Convergence")
plt.tight_layout()
plt.savefig("bnn_loss.png", dpi=1200)

# 使用 Monte Carlo 方法生成仿真数据
bnn_model.eval()
n_samples = 1000
sim_X = torch.tensor(np.random.normal(0, 1, size=(n_samples, 3)), dtype=torch.float32)
sim_y_samples = [bnn_model(sim_X).detach().numpy() for _ in range(100)]
sim_y_stack = np.stack(sim_y_samples, axis=0)
sim_y_mean = sim_y_stack.mean(axis=0)
sim_y_std = sim_y_stack.std(axis=0)

# 保存仿真数据
np.savez("simulated_data.npz", X=sim_X.numpy(), y=sim_y_mean)

# 可视化仿真数据不确定性

samples_per_plot = 250
total_samples = 1000
num_plots = total_samples // samples_per_plot

for i in range(num_plots):
    start = i * samples_per_plot
    end = start + samples_per_plot

    x = np.arange(start, end)
    y_mean = sim_y_mean[start:end, 0]   # output1
    y_std  = sim_y_std[start:end, 0]    # 不确定性为标准差

    plt.figure(figsize=(8, 4))
    plt.errorbar(x, y_mean, yerr=y_std, fmt='o', ecolor='red', alpha=0.6, capsize=3)
    plt.title(f"Prediction with Uncertainty (Output1) - Samples {start} to {end}")
    plt.xlabel("Sample Index")
    plt.ylim(-2, 6)
    plt.ylabel("Prediction ± Std")
    plt.tight_layout()
    plt.savefig(f"uncertainty_output1_part{i+1}.png", dpi=300)
    plt.close()
    

for i in range(num_plots):
    start = i * samples_per_plot
    end = start + samples_per_plot

    x = np.arange(start, end)
    y_mean = sim_y_mean[start:end, 1]   # output2
    y_std  = sim_y_std[start:end, 1]    # 不确定性为标准差

    plt.figure(figsize=(8, 4))
    plt.errorbar(x, y_mean, yerr=y_std, fmt='o', ecolor='red', alpha=0.6, capsize=3)
    plt.title(f"Prediction with Uncertainty (Output2) - Samples {start} to {end}")
    plt.xlabel("Sample Index")
    plt.ylim(-2, 6)
    plt.ylabel("Prediction ± Std")
    plt.tight_layout()
    plt.savefig(f"uncertainty_output2_part{i+1}.png", dpi=300)
    plt.close()

# 加载仿真数据
sim_data = np.load("simulated_data.npz")
X_sim = torch.tensor(sim_data["X"], dtype=torch.float32)
y_sim = torch.tensor(sim_data["y"], dtype=torch.float32)



# BNN 模型训练 
bnn_model_sim = BayesianNN(input_dim=3, output_dim=2)
bnn_loss_sim = train_bnn(bnn_model_sim, X_sim, y_sim)

# 模型保存
torch.save(bnn_model_sim.state_dict(), "bnn_trained.pth")

# BNN 损失收敛图 
def plot_loss_curve(loss_list, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
plot_loss_curve(bnn_loss_sim, "BNN Loss (Simulated Data)", "bnn_sim_loss.png")


# ANN 模型训练仿真数据
class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        '''self.dropout = nn.Dropout(p=0.1)'''
        self.out = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        '''x = self.dropout(x)'''
        x = F.relu(self.fc2(x))
        '''x = self.dropout(x)'''
        x = F.relu(self.fc3(x))
        '''x = self.dropout(x)'''
        return self.out(x)

def train_ann(model, X_train, y_train, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return loss_history

# ANN 模型训练
ann_model_sim = ANN(input_dim=3, output_dim=2)
ann_loss_sim = train_ann(ann_model_sim, X_sim, y_sim)
plot_loss_curve(ann_loss_sim, "ANN Loss (Simulated Data)", "ann_sim_loss.png")

# 模型保存
torch.save(ann_model_sim.state_dict(), "ann_trained.pth")


# 同图绘制 ANN 和 BNN 对比 Loss 
plt.figure(figsize=(8, 5))
plt.plot(bnn_loss_sim, label="BNN Loss")
plt.plot(ann_loss_sim, label="ANN Loss")
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.title("Loss Comparison (Simulated Data)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_comparison_sim.png", dpi=300)
plt.close()


# 反标准化
def inverse_transform_y(tensor_y):
    return scaler_y.inverse_transform(tensor_y.detach().cpu().numpy())

# 模型评估与对比
with torch.no_grad():
    bnn_real_preds = bnn_model(X_test)
    bnn_sim_preds = bnn_model_sim(X_test)
    ann_sim_preds = ann_model_sim(X_test)

bnn_true = inverse_transform_y(y_test)
bnn_real_pred = inverse_transform_y(bnn_real_preds)
bnn_sim_pred = inverse_transform_y(bnn_sim_preds)
ann_sim_pred = inverse_transform_y(ann_sim_preds)

'''def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    with open(f"{name}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\n")
    return mse, mae, r2

evaluate("BNN_real", bnn_true, bnn_real_pred)
evaluate("BNN_sim", bnn_true, bnn_sim_pred)
evaluate("ANN_sim", bnn_true, ann_sim_pred)'''

def evaluate(name, y_true, y_pred):
    results = {}
    with open(f"{name}_metrics.txt", "w", encoding="utf-8") as f:
        for i in range(y_true.shape[1]):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            results[f"Output{i}"] = {"MSE": mse, "MAE": mae, "R2": r2}
            f.write(f"Output{i}:\n")
            f.write(f"  MSE: {mse:.4f}\n")
            f.write(f"  MAE: {mae:.4f}\n")
            f.write(f"  R²:  {r2:.4f}\n\n")
    return results

bnn_real_metrics = evaluate("BNN_real", bnn_true, bnn_real_pred)
bnn_sim_metrics  = evaluate("BNN_sim", bnn_true, bnn_sim_pred)
ann_sim_metrics  = evaluate("ANN_sim", bnn_true, ann_sim_pred)

# 可视化预测 vs 真实值
for i in range(2):
    plt.figure()
    plt.scatter(bnn_true[:, i], bnn_sim_pred[:, i], label="BNN", alpha=0.6)
    plt.scatter(bnn_true[:, i], ann_sim_pred[:, i], label="ANN", alpha=0.6)
    plt.plot([bnn_true[:, i].min(), bnn_true[:, i].max()],
             [bnn_true[:, i].min(), bnn_true[:, i].max()], 'k--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Output {i+1}: Predicted vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pred_vs_true_output{i+1}.png", dpi=300)

# 可视化仿真数据散点图（反标准化）
sim_X_inv = scaler_X.inverse_transform(sim_X.numpy())
sim_y_inv = scaler_y.inverse_transform(sim_y_mean)
sim_X_inv = np.clip(sim_X_inv, a_min=1e-6, a_max=None)  
sim_y_inv = np.clip(sim_y_inv, a_min=1e-6, a_max=None) 


plt.figure(figsize=(8, 6))
plt.scatter(y_raw[:, 0], y_raw[:, 1], c='blue', label='Real', alpha=0.7)
plt.scatter(sim_y_inv[:, 0], sim_y_inv[:, 1], c='orange', label='Simulated', alpha=0.3)
plt.xlabel("Output 1"), plt.ylabel("Output 2"), plt.legend()
plt.title("real_data vs sim_data")
plt.tight_layout()
plt.savefig("real_vs_simulated.png", dpi = 300)
plt.show()


# 绘制固定第三特征时的 Heatmap
fixed_val = 0.0  # 第 3 个特征固定值
f1 = np.linspace(-2, 2, 50)
f2 = np.linspace(-2, 2, 50)
F1, F2 = np.meshgrid(f1, f2)
F3 = np.full_like(F1, fixed_val)
grid_input = np.stack([F1.ravel(), F2.ravel(), F3.ravel()], axis=1)
grid_tensor = torch.tensor(grid_input, dtype=torch.float32)

# 生成 BNN 预测的 Heatmap
bnn_model_sim.eval()
heatmap_preds = [bnn_model_sim(grid_tensor).detach().numpy() for _ in range(100)]
heatmap_mean = np.mean(heatmap_preds, axis=0)
output1_map = heatmap_mean[:, 0].reshape(F1.shape)
output2_map = heatmap_mean[:, 1].reshape(F1.shape)

# 输出 Output1 热图
plt.figure(figsize=(6, 5))
sns.heatmap(output1_map, xticklabels=False, yticklabels=False, cmap="YlGnBu")
plt.title("BNN Output1 Heatmap (fixed Time)")
plt.tight_layout()
plt.savefig("BNN heatmap_output1_fixed_t.png", dpi=300)

# 输出 Output2 热图
plt.figure(figsize=(6, 5))
sns.heatmap(output2_map, xticklabels=False, yticklabels=False, cmap="YlOrRd")
plt.title("BNN Output2 Heatmap (fixed Time)")
plt.tight_layout()
plt.savefig("BNN heatmap_output2_fixed_t.png", dpi=300)

# 生成 ANN 预测的 Heatmap
ann_model_sim.eval()
heatmap_preds = [ann_model_sim(grid_tensor).detach().numpy() for _ in range(30)]
heatmap_mean = np.mean(heatmap_preds, axis=0)
output1_map = heatmap_mean[:, 0].reshape(F1.shape)
output2_map = heatmap_mean[:, 1].reshape(F1.shape)

# 输出 Output1 热图
plt.figure(figsize=(6, 5))
sns.heatmap(output1_map, xticklabels=False, yticklabels=False, cmap="YlGnBu")
plt.title("ANN Output1 Heatmap (fixed Time)")
plt.tight_layout()
plt.savefig("ANN heatmap_output1_fixed_t.png", dpi=300)

# 输出 Output2 热图
plt.figure(figsize=(6, 5))
sns.heatmap(output2_map, xticklabels=False, yticklabels=False, cmap="YlOrRd")
plt.title("ANN Output2 Heatmap (fixed Time)")
plt.tight_layout()
plt.savefig("ANN heatmap_output2_fixed_t.png", dpi=300)

# BNN Monte Carlo预测 
bnn_model.eval()
num_samples = 1000

# 多次预测，得到预测分布
heatmap_preds_std = np.stack([
    bnn_model(grid_tensor).detach().numpy() for _ in range(num_samples)
], axis=0)  # shape: (1000, N, 2)

# 计算均值和标准差（仍为标准化后的值）
mean_preds_std = np.mean(heatmap_preds_std, axis=0)  # shape: (N, 2)
std_preds_std = np.std(heatmap_preds_std, axis=0)    # shape: (N, 2)

# 反标准化 
mean_preds = scaler_y.inverse_transform(mean_preds_std)
std_preds = std_preds_std * scaler_y.scale_  

# 分别提取 Output1 和 Output2 的均值和不确定性
mean_output1 = mean_preds[:, 0].reshape(F1.shape)
mean_output2 = mean_preds[:, 1].reshape(F1.shape)
std_output1 = std_preds[:, 0].reshape(F1.shape)
std_output2 = std_preds[:, 1].reshape(F1.shape)

# 可视化均值 
plt.figure(figsize=(6, 5))
sns.heatmap(mean_output1, xticklabels=False, yticklabels=False, cmap="YlGnBu")
plt.title("BNN Mean Output1 (rev_std)")
plt.tight_layout()
plt.savefig("BNN_mean_output1_realvalue.png", dpi=300)

plt.figure(figsize=(6, 5))
sns.heatmap(mean_output2, xticklabels=False, yticklabels=False, cmap="YlOrRd")
plt.title("BNN Mean Output2 (rev_std)")
plt.tight_layout()
plt.savefig("BNN_mean_output2_realvalue.png", dpi=300)

# 可视化不确定性（标准差） 
plt.figure(figsize=(6, 5))
sns.heatmap(std_output1, xticklabels=False, yticklabels=False, cmap="Purples")
plt.title("BNN Output1 uncertainty (Std. Dev)")
plt.tight_layout()
plt.savefig("BNN_uncertainty_output1.png", dpi=300)

plt.figure(figsize=(6, 5))
sns.heatmap(std_output2, xticklabels=False, yticklabels=False, cmap="Oranges")
plt.title("BNN Output2 uncertainty (Std. Dev)")
plt.tight_layout()
plt.savefig("BNN_uncertainty_output2.png", dpi=300)
