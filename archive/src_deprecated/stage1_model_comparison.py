#!/usr/bin/env python3
"""
第一阶段：四模型对比预测传播势能 P(t)

模型:
1. TCN (Temporal Convolutional Network)
2. LSTM (Long Short-Term Memory)
3. GRU (Gated Recurrent Unit)
4. MLP (Multi-Layer Perceptron) - baseline

输入: 气象序列 [T, H, P]
输出: 传播势能 P(t)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 70)
log("第一阶段：四模型对比预测传播势能 P(t)")
log("=" * 70)

# ============================================================
# 1. 温度依赖的传播势能 (目标变量)
# ============================================================
def calc_transmission_potential(T):
    """计算传播势能 P(T) = a²×b×c/μ_m"""
    if T < 15 or T > 35:
        return 0.01
    
    a = max(0.01, 0.0005 * T * (T - 14) * np.sqrt(max(0.01, 35 - T)))
    b = max(0.01, 0.0008 * T * (T - 17) * np.sqrt(max(0.01, 36 - T)))
    c = max(0.01, 0.0007 * T * (T - 12) * np.sqrt(max(0.01, 37 - T)))
    mu_m = max(0.02, 0.0006 * T**2 - 0.028 * T + 0.37)
    
    return (a ** 2) * b * c / mu_m

# ============================================================
# 2. 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2005) & (case_df['year'] <= 2019)].copy()

log(f"  数据范围: {df['year'].min()}-{df['year'].max()}")
log(f"  样本数: {len(df)} 个月")

# 特征和目标
features = df[['temperature', 'humidity', 'precipitation']].values
P_target = np.array([calc_transmission_potential(T) for T in df['temperature'].values])

log(f"  特征: temperature, humidity, precipitation")
log(f"  目标: P(T) 传播势能")
log(f"  P(T) 范围: {P_target.min():.4f} - {P_target.max():.4f}")

# ============================================================
# 3. 数据预处理
# ============================================================
log("\n[2] 数据预处理...")

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(P_target.reshape(-1, 1)).flatten()

# 创建序列数据
SEQ_LEN = 6  # 用过去6个月预测当前

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

log(f"  序列长度: {SEQ_LEN}")
log(f"  序列样本数: {len(X_seq)}")

# 划分训练/验证/测试
n_total = len(X_seq)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

X_train = X_seq[:n_train]
y_train = y_seq[:n_train]
X_val = X_seq[n_train:n_train+n_val]
y_val = y_seq[n_train:n_train+n_val]
X_test = X_seq[n_train+n_val:]
y_test = y_seq[n_train+n_val:]

log(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# 转换为Tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# ============================================================
# 4. 模型定义
# ============================================================

# 4.1 TCN
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        
    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size=3, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.fc(out).squeeze(-1)

# 4.2 LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

# 4.3 GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

# 4.4 MLP (baseline)
class MLPModel(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_size * seq_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x).squeeze(-1)

# ============================================================
# 5. 训练函数
# ============================================================
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.01, patience=20):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_state)
    return model, best_val_loss

def evaluate_model(model, X_test, y_test, scaler_y):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
    
    # 反标准化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    
    # 指标
    corr, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'corr': corr,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_pred,
        'y_true': y_true
    }

# ============================================================
# 6. 训练四个模型
# ============================================================
log("\n[3] 训练四个模型...")

INPUT_SIZE = 3  # temperature, humidity, precipitation
HIDDEN_SIZE = 32

models = {
    'TCN': TCNModel(INPUT_SIZE, HIDDEN_SIZE),
    'LSTM': LSTMModel(INPUT_SIZE, HIDDEN_SIZE),
    'GRU': GRUModel(INPUT_SIZE, HIDDEN_SIZE),
    'MLP': MLPModel(INPUT_SIZE, SEQ_LEN, HIDDEN_SIZE * 2)
}

results = {}

for name, model in models.items():
    log(f"\n  训练 {name}...")
    
    # 训练
    model, val_loss = train_model(model, X_train_t, y_train_t, X_val_t, y_val_t)
    
    # 评估
    metrics = evaluate_model(model, X_test_t, y_test_t, scaler_y)
    metrics['val_loss'] = val_loss
    metrics['model'] = model
    results[name] = metrics
    
    log(f"    验证损失: {val_loss:.6f}")
    log(f"    测试集 - r: {metrics['corr']:.4f}, R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

# ============================================================
# 7. 模型对比
# ============================================================
log("\n" + "=" * 70)
log("模型对比结果")
log("=" * 70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Correlation': [results[m]['corr'] for m in results],
    'R²': [results[m]['r2'] for m in results],
    'MAE': [results[m]['mae'] for m in results],
    'RMSE': [results[m]['rmse'] for m in results]
})

log("\n" + comparison_df.to_string(index=False))

# 选择最佳模型
best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
best_model = results[best_model_name]['model']
best_metrics = results[best_model_name]

log(f"\n最佳模型: {best_model_name}")
log(f"  相关系数: {best_metrics['corr']:.4f}")
log(f"  R²: {best_metrics['r2']:.4f}")

# ============================================================
# 8. 用最佳模型预测全量数据
# ============================================================
log("\n[4] 用最佳模型预测全量数据...")

# 全量预测
X_all_t = torch.FloatTensor(X_seq)
best_model.eval()
with torch.no_grad():
    y_pred_all_scaled = best_model(X_all_t).numpy()

y_pred_all = scaler_y.inverse_transform(y_pred_all_scaled.reshape(-1, 1)).flatten()
y_true_all = P_target[SEQ_LEN:]

# 全量评估
corr_all, _ = pearsonr(y_true_all, y_pred_all)
r2_all = r2_score(y_true_all, y_pred_all)

log(f"  全量数据 - r: {corr_all:.4f}, R²: {r2_all:.4f}")

# ============================================================
# 9. 可视化
# ============================================================
log("\n[5] 生成可视化...")

fig = plt.figure(figsize=(18, 12))

# 1. 模型对比柱状图
ax1 = fig.add_subplot(2, 3, 1)
x = range(len(comparison_df))
width = 0.35
ax1.bar([i - width/2 for i in x], comparison_df['Correlation'], width, label='Correlation', color='steelblue')
ax1.bar([i + width/2 for i in x], comparison_df['R²'], width, label='R²', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['Model'])
ax1.set_ylabel('Score')
ax1.set_title('Model Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 最佳模型预测 vs 实际
ax2 = fig.add_subplot(2, 3, 2)
months = range(len(y_true_all))
ax2.plot(months, y_true_all, 'b-', lw=1.5, label='True P(T)', alpha=0.8)
ax2.plot(months, y_pred_all, 'r-', lw=1.5, label=f'{best_model_name} Predicted', alpha=0.8)
ax2.set_xlabel('Month')
ax2.set_ylabel('Transmission Potential P(T)')
ax2.set_title(f'{best_model_name}: Prediction (r={corr_all:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 散点图
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(y_true_all, y_pred_all, alpha=0.6, s=30)
max_val = max(y_true_all.max(), y_pred_all.max())
ax3.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax3.set_xlabel('True P(T)')
ax3.set_ylabel('Predicted P(T)')
ax3.set_title(f'Scatter (R²={r2_all:.3f})')
ax3.grid(True, alpha=0.3)

# 4-7. 各模型测试集预测
for idx, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(2, 3, idx + 4) if idx < 2 else None
    if ax is None:
        break
    ax.plot(res['y_true'], 'b-', lw=1, label='True', alpha=0.7)
    ax.plot(res['y_pred'], 'r-', lw=1, label='Pred', alpha=0.7)
    ax.set_title(f"{name} (r={res['corr']:.3f}, R²={res['r2']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

log("  已保存: results/figures/model_comparison.png")

# ============================================================
# 10. 保存结果供符号回归使用
# ============================================================
log("\n[6] 保存结果供符号回归...")

# 准备符号回归数据
# 输入: 气象特征, 输出: P(T)预测值
sr_data = pd.DataFrame({
    'temperature': df['temperature'].values[SEQ_LEN:],
    'humidity': df['humidity'].values[SEQ_LEN:],
    'precipitation': df['precipitation'].values[SEQ_LEN:],
    'P_true': y_true_all,
    'P_predicted': y_pred_all
})

sr_data.to_csv('/root/wenmei/results/data/for_symbolic_regression.csv', index=False)
log("  已保存: results/data/for_symbolic_regression.csv")

# 保存模型对比结果
comparison_df.to_csv('/root/wenmei/results/data/model_comparison.csv', index=False)
log("  已保存: results/data/model_comparison.csv")

# 保存最佳模型
torch.save(best_model.state_dict(), '/root/wenmei/results/data/best_model.pt')
log(f"  已保存: results/data/best_model.pt ({best_model_name})")

# ============================================================
# 11. 总结
# ============================================================
log("\n" + "=" * 70)
log("第一阶段总结")
log("=" * 70)
log(f"""
【任务】预测传播势能 P(T)
【输入】气象序列 (T, H, P), 序列长度={SEQ_LEN}
【目标】P(T) = a²×b×c/μ_m

【模型对比】
{comparison_df.to_string(index=False)}

【最佳模型】{best_model_name}
  相关系数: {best_metrics['corr']:.4f}
  R²: {best_metrics['r2']:.4f}
  RMSE: {best_metrics['rmse']:.4f}

【输出文件】
  - results/data/for_symbolic_regression.csv (符号回归输入)
  - results/data/model_comparison.csv (对比结果)
  - results/data/best_model.pt (最佳模型权重)

【下一步】符号回归寻找P(T)的解析表达式
""")

log("\n第一阶段完成!")
