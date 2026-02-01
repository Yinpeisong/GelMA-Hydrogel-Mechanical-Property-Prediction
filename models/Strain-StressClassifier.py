import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

plt.rc('font', family='Arial')   #设置全局字体为Arial
plt.rcParams.update({'font.size': 18,  'font.weight': 'bold'})     #设置全局字体大小为16

# 配置参数 
data_path = "data/CriticalStress.csv"  # CSV路径
save_model_path = "models/bnn_model_classifier.pth"
num_epochs = 1000
batch_size = 32
learning_rate = 1e-3
dropout_rate = 0.1
n_classes = 3
monte_carlo_samples = 180  # 用于不确定性估计

# 加载与标准化数据 
data = pd.read_csv(data_path)
X = data.iloc[:, :-1].values  # 前3列为特征
y = data.iloc[:, -1].values   # 最后一列为标签

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "models/classifier_scaler_X.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义 BNN 模型 
class BNNClassifier(nn.Module):
    def __init__(self, input_dim,  output_dim, dropout_rate=0.1):
        super(BNNClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=32)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(in_features=16, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

model = BNNClassifier(input_dim = 3, output_dim=n_classes, dropout_rate=dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练 
train_loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_loss_history.append(avg_loss)
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 保存模型和标准化器
torch.save(model.state_dict(), save_model_path)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Monte Carlo Dropout 不确定性分析
model.eval()

def predict_mc(model, x, samples=180):
    model.train()  # Dropout only active in training mode
    probs = []
    for _ in range(samples):
        logits = model(x)
        prob = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(prob)
    probs = np.array(probs)  # shape: [samples, batch, classes]
    return probs.mean(axis=0), probs.std(axis=0)

mean_probs, std_probs = predict_mc(model, X_test_tensor, samples=monte_carlo_samples)
y_pred = np.argmax(mean_probs, axis=1)
uncertainty = std_probs.mean(axis=1)

# 可视化与评估 
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("CriticalStressOutput/confusion_matrix.png")

# 不确定性可视化
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), uncertainty, c=y_pred, cmap='viridis', alpha=0.7)
plt.colorbar(label="Predicted Class")
plt.title("Uncertainty (MC Dropout Std Dev)")
plt.xlabel("Sample Index")
plt.ylabel("Uncertainty")
plt.tight_layout()
plt.savefig("CriticalStressOutput/uncertainty_plot.png")

# 损失函数曲线
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("CriticalStressOutput/loss_curve.png")
