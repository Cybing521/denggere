"""
混合SEI-SEIR + TCN模型：直接预测病例数

方法：
1. 使用气象数据和BI数据作为特征
2. TCN直接预测月度病例数
3. SEI-SEIR提供物理约束（R0作为辅助特征）
4. 滞后特征捕捉传播延迟

时间尺度：月度（Monthly）
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


class CasePredictionTCN(nn.Module):
    """TCN模型直接预测病例数"""
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.3):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            padding = dilation
            
            layers.append(nn.Conv1d(in_ch, hidden_size, kernel_size=3, 
                                   padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x[:, :, -1]  # 最后时间步
        return self.fc(x)


def compute_seir_features(row, prev_cases=0):
    """
    计算SEI-SEIR相关特征
    """
    temp = row['temperature']
    humid = row['humidity']
    precip = row['precipitation']
    
    # 温度适宜性 (最适25-28°C)
    T_opt, T_w = 27, 6
    temp_suitability = np.exp(-((temp - T_opt) / T_w)**2)
    
    # 湿度适宜性 (最适70-85%)
    H_opt, H_w = 77, 15
    humid_suitability = np.exp(-((humid - H_opt) / H_w)**2)
    
    # 降水因子
    precip_factor = np.tanh(precip / 4)
    
    # 综合环境适宜性
    env_suitability = temp_suitability * humid_suitability * (0.3 + 0.7 * precip_factor)
    
    # 简化R0估计
    base_R0 = 0.5 + 2.0 * env_suitability
    
    # 季节因子 (月份)
    month = row['month']
    season_factor = np.sin(np.pi * (month - 3) / 6)  # 夏季高
    
    return {
        'temp_suit': temp_suitability,
        'humid_suit': humid_suitability,
        'precip_factor': precip_factor,
        'env_suit': env_suitability,
        'est_R0': base_R0,
        'season_factor': season_factor
    }


def prepare_features(df, bi_df, seq_length=6):
    """
    准备特征数据
    """
    # 合并BI数据
    merged = pd.merge(df, bi_df, on=['year', 'month'], how='left')
    merged['bi'] = merged['bi'].fillna(merged['bi'].mean())
    
    # 计算SEI-SEIR特征
    seir_features = merged.apply(compute_seir_features, axis=1)
    seir_df = pd.DataFrame(seir_features.tolist())
    
    for col in seir_df.columns:
        merged[col] = seir_df[col].values
    
    # 添加滞后特征
    for lag in [1, 2, 3]:
        merged[f'cases_lag{lag}'] = merged['cases'].shift(lag).fillna(0)
        merged[f'bi_lag{lag}'] = merged['bi'].shift(lag).fillna(merged['bi'].mean())
        merged[f'env_suit_lag{lag}'] = merged['env_suit'].shift(lag).fillna(merged['env_suit'].mean())
    
    # 特征列表
    feature_cols = [
        'temperature', 'humidity', 'precipitation',
        'bi', 'temp_suit', 'humid_suit', 'precip_factor',
        'env_suit', 'est_R0', 'season_factor',
        'cases_lag1', 'cases_lag2', 'cases_lag3',
        'bi_lag1', 'bi_lag2',
        'env_suit_lag1', 'env_suit_lag2'
    ]
    
    X = merged[feature_cols].values
    y = merged['cases'].values
    
    # 归一化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # 创建序列数据
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_length:i])
        y_seq.append(y_scaled[i])
    
    return (np.array(X_seq), np.array(y_seq), 
            scaler_X, scaler_y, merged, feature_cols)


def train_model(X, y, epochs=300, lr=0.001, val_split=0.2):
    """
    训练TCN模型
    """
    # 转换为tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 划分训练集/验证集
    n_train = int(len(X) * (1 - val_split))
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]
    
    # 初始化模型
    model = CasePredictionTCN(
        input_size=X.shape[2],
        hidden_size=64,
        num_layers=3,
        dropout=0.3
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 50:
            print(f"  早停于 epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")
    
    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_and_visualize(model, X, y, scaler_y, merged_df, seq_length, save_path):
    """
    评估模型并可视化
    """
    model.eval()
    
    # 预测
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
    
    # 反归一化
    predicted = scaler_y.inverse_transform(pred_scaled).flatten()
    predicted = np.maximum(predicted, 0)  # 确保非负
    
    actual = merged_df['cases'].values[seq_length:]
    years = merged_df['year'].values[seq_length:]
    months = merged_df['month'].values[seq_length:]
    
    # 评估指标
    r2 = r2_score(actual, predicted)
    log_r2 = r2_score(np.log1p(actual), np.log1p(predicted))
    corr, pval = pearsonr(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    print(f"\n  模型评估结果:")
    print(f"    R² (线性): {r2:.4f}")
    print(f"    R² (对数): {log_r2:.4f}")
    print(f"    相关系数: {corr:.4f} (p={pval:.2e})")
    print(f"    MAE: {mae:.1f}")
    print(f"    RMSE: {rmse:.1f}")
    
    # 可视化
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 时间序列对比
    ax1 = fig.add_subplot(2, 3, 1)
    time_idx = range(len(actual))
    ax1.plot(time_idx, actual, 'b-', lw=2, label='Actual', marker='o', ms=3, alpha=0.7)
    ax1.plot(time_idx, predicted, 'r-', lw=2, label='Predicted', marker='s', ms=3, alpha=0.7)
    ax1.set_xlabel('Month Index')
    ax1.set_ylabel('Cases')
    ax1.set_title('Monthly Dengue Cases: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 对数尺度
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(time_idx, actual + 1, 'b-', lw=2, label='Actual', marker='o', ms=3)
    ax2.semilogy(time_idx, predicted + 1, 'r-', lw=2, label='Predicted', marker='s', ms=3)
    ax2.set_xlabel('Month Index')
    ax2.set_ylabel('Cases (log scale)')
    ax2.set_title('Cases Comparison (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 散点图
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(actual, predicted, alpha=0.6, c='steelblue', s=50, edgecolors='white')
    max_val = max(actual.max(), predicted.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Cases')
    ax3.set_ylabel('Predicted Cases')
    ax3.set_title(f'Prediction vs Actual\nR²={r2:.3f}, r={corr:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 年度对比
    ax4 = fig.add_subplot(2, 3, 4)
    df_eval = pd.DataFrame({
        'year': years, 'actual': actual, 'predicted': predicted
    })
    yearly = df_eval.groupby('year').agg({'actual': 'sum', 'predicted': 'sum'}).reset_index()
    x = range(len(yearly))
    width = 0.35
    ax4.bar([i - width/2 for i in x], yearly['actual'], width, label='Actual', color='steelblue')
    ax4.bar([i + width/2 for i in x], yearly['predicted'], width, label='Predicted', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(yearly['year'])
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Annual Cases')
    ax4.set_title('Annual Cases: Actual vs Predicted')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 2014年详情
    ax5 = fig.add_subplot(2, 3, 5)
    mask_2014 = years == 2014
    if mask_2014.sum() > 0:
        actual_2014 = actual[mask_2014]
        pred_2014 = predicted[mask_2014]
        months_2014 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(actual_2014)]
        x_2014 = range(len(actual_2014))
        ax5.bar([i - 0.2 for i in x_2014], actual_2014, 0.4, label='Actual', color='red', alpha=0.7)
        ax5.bar([i + 0.2 for i in x_2014], pred_2014, 0.4, label='Predicted', color='orange', alpha=0.7)
        ax5.set_xticks(x_2014)
        ax5.set_xticklabels(months_2014, rotation=45)
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Cases')
        ax5.set_title('2014 Outbreak: Monthly Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. 性能总结
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
    Hybrid SEI-SEIR + TCN Model
    ===========================
    
    Data: Guangdong Province
    Period: 2006-2014
    Time Scale: Monthly
    Samples: {len(actual)} months
    
    Performance Metrics:
    - R² (linear): {r2:.4f}
    - R² (log): {log_r2:.4f}
    - Correlation: {corr:.4f}
    - MAE: {mae:.0f} cases
    - RMSE: {rmse:.0f} cases
    
    2014 Outbreak:
    - Actual Peak: {actual[mask_2014].max():.0f}
    - Predicted Peak: {predicted[mask_2014].max():.0f}
    - Annual Actual: {actual[mask_2014].sum():.0f}
    - Annual Predicted: {predicted[mask_2014].sum():.0f}
    
    Model Features:
    - Meteorological data
    - Breteau Index (BI)
    - SEI-SEIR derived features
    - Lagged case counts (1-3 months)
    """
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'actual': actual,
        'predicted': predicted,
        'years': years,
        'months': months,
        'r2': r2,
        'log_r2': log_r2,
        'corr': corr,
        'mae': mae,
        'rmse': rmse
    }


def main():
    print("=" * 70)
    print("混合SEI-SEIR + TCN模型：直接预测病例数")
    print("时间尺度: 月度 (Monthly)")
    print("=" * 70)
    
    # 加载数据
    print("\n[1] 加载数据...")
    case_df = pd.read_csv('../data/guangdong_dengue_cases.csv')
    case_df['date'] = pd.to_datetime(case_df['date'])
    
    bi_df = pd.read_csv('../data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    
    # 分析时段
    analysis_df = case_df[(case_df['year'] >= 2006) & (case_df['year'] <= 2014)].copy()
    print(f"  数据时段: 2006-2014年")
    print(f"  月份数: {len(analysis_df)}")
    print(f"  总病例: {analysis_df['cases'].sum():,}")
    
    # 准备特征
    print("\n[2] 准备特征 (气象 + BI + SEI-SEIR特征 + 滞后)...")
    seq_length = 6
    X, y, scaler_X, scaler_y, merged_df, feature_cols = prepare_features(
        analysis_df, gz_bi, seq_length=seq_length
    )
    print(f"  特征数: {len(feature_cols)}")
    print(f"  序列长度: {seq_length}个月")
    print(f"  训练样本: {len(X)}")
    
    # 训练模型
    print("\n[3] 训练TCN模型...")
    model, train_losses, val_losses = train_model(X, y, epochs=300, lr=0.002)
    
    # 评估和可视化
    print("\n[4] 模型评估...")
    save_path = '../results/hybrid_model_results.png'
    results = evaluate_and_visualize(
        model, X, y, scaler_y, merged_df, seq_length, save_path
    )
    print(f"\n  已保存: {save_path}")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'year': results['years'],
        'month': results['months'],
        'actual_cases': results['actual'],
        'predicted_cases': results['predicted']
    })
    results_df.to_csv('../results/hybrid_model_predictions.csv', index=False)
    print(f"  已保存: ../results/hybrid_model_predictions.csv")
    
    print("\n" + "=" * 70)
    print("模型训练完成!")
    print("=" * 70)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
