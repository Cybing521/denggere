"""
两阶段SEI-SEIR模型 V2：改进的病例拟合

改进点：
1. 使用对数尺度拟合（处理跨数量级变化）
2. 分段校准（低/高病例期）
3. 更合理的参数约束
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


class SimpleTCN(nn.Module):
    """简化的TCN模型"""
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=2)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x[:, :, -1]  # 最后时间步
        return self.fc(x)


class ImprovedSEISEIR:
    """改进的SEI-SEIR模拟器"""
    
    def __init__(self, N_h=110000000):
        self.N_h = N_h
        
        # 固定生物学参数
        self.mu_v = 1/14       # 蚊虫死亡率
        self.sigma_v = 1/10    # 蚊虫EIP
        self.sigma_h = 1/5.5   # 人潜伏期
        self.gamma_h = 1/7     # 人恢复率
        
    def simulate(self, lambda_v_series, params):
        """
        模拟月度新增病例
        
        params: [beta_vh, beta_hv, b, m_ratio, init_I_v_frac, case_scale]
        """
        beta_vh, beta_hv, b, m_ratio, init_I_v_frac, case_scale = params
        
        months = len(lambda_v_series)
        
        # 初始条件
        N_v = self.N_h * m_ratio
        I_v_init = N_v * init_I_v_frac
        S_v = N_v - I_v_init
        E_v = I_v_init * 0.5
        I_v = I_v_init * 0.5
        
        S_h = self.N_h
        E_h = 0
        I_h = 1  # 初始1个感染者
        R_h = 0
        
        new_cases_monthly = []
        R0_series = []
        
        dt = 0.5  # 半天步长，提高精度
        steps_per_month = int(30 / dt)
        
        for m in range(months):
            lambda_v = lambda_v_series[m]
            monthly_new = 0
            
            for step in range(steps_per_month):
                # 传播力
                force_hv = b * beta_hv * I_h / self.N_h if I_h > 0 else 0
                force_vh = b * beta_vh * I_v / self.N_h if I_v > 0 else 0
                
                # 蚊虫动态
                dS_v = lambda_v * dt - force_hv * S_v * dt - self.mu_v * S_v * dt
                dE_v = force_hv * S_v * dt - (self.sigma_v + self.mu_v) * E_v * dt
                dI_v = self.sigma_v * E_v * dt - self.mu_v * I_v * dt
                
                # 人群动态
                new_infections = self.sigma_h * E_h * dt
                dS_h = -force_vh * S_h * dt
                dE_h = force_vh * S_h * dt - self.sigma_h * E_h * dt
                dI_h = new_infections - self.gamma_h * I_h * dt
                dR_h = self.gamma_h * I_h * dt
                
                # 更新
                S_v = max(0, S_v + dS_v)
                E_v = max(0, E_v + dE_v)
                I_v = max(0, I_v + dI_v)
                S_h = max(0, S_h + dS_h)
                E_h = max(0, E_h + dE_h)
                I_h = max(0, I_h + dI_h)
                R_h = max(0, R_h + dR_h)
                
                monthly_new += new_infections
            
            new_cases_monthly.append(monthly_new * case_scale)
            
            # R0计算
            N_v_cur = S_v + E_v + I_v
            m_cur = N_v_cur / self.N_h
            R0 = (b**2 * beta_vh * beta_hv * m_cur * self.sigma_v) / \
                 ((self.sigma_v + self.mu_v) * self.mu_v * self.gamma_h)
            R0_series.append(R0)
        
        return np.array(new_cases_monthly), np.array(R0_series)


def train_tcn_for_lambda(weather_df, bi_df):
    """训练TCN预测λ_v"""
    # 合并数据
    merged = pd.merge(weather_df, bi_df, on=['year', 'month'], how='left')
    merged = merged.dropna(subset=['bi'])
    
    # 准备数据
    features = ['temperature', 'humidity', 'precipitation']
    X = merged[features].values
    y = merged['bi'].values
    
    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # 创建序列
    seq_len = 4
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_len:i])
        y_seq.append(y_scaled[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 训练
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq)
    
    model = SimpleTCN(input_size=3, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model, scaler_X, scaler_y, seq_len


def predict_lambda_v(model, scaler_X, scaler_y, weather_df, seq_len):
    """预测λ_v序列"""
    model.eval()
    
    features = ['temperature', 'humidity', 'precipitation']
    X = weather_df[features].values
    X_scaled = scaler_X.transform(X)
    
    lambda_v = []
    
    for i in range(len(X)):
        temp, humid, precip = X[i]
        
        # 基于气象的简单λ_v估计
        T_opt, T_w = 27, 6
        H_opt, H_w = 78, 15
        
        temp_f = np.exp(-((temp - T_opt) / T_w)**2)
        humid_f = np.exp(-((humid - H_opt) / H_w)**2)
        precip_f = np.tanh(precip / 4)
        
        # 基础λ_v
        base_lambda = 8000 * temp_f * humid_f * (0.4 + 0.6 * precip_f)
        
        # TCN调整
        if i >= seq_len:
            seq = X_scaled[i-seq_len:i]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
            with torch.no_grad():
                bi_pred = model(seq_tensor).item()
            bi_pred = scaler_y.inverse_transform([[bi_pred]])[0, 0]
            # BI修正λ_v
            bi_factor = 1 + 0.1 * max(0, bi_pred - 5)  # BI>5时增强
            base_lambda *= bi_factor
        
        lambda_v.append(max(500, base_lambda))
    
    return np.array(lambda_v)


def calibrate_params(lambda_v, actual_cases):
    """校准SEI-SEIR参数"""
    simulator = ImprovedSEISEIR()
    
    # 对数变换处理跨数量级数据
    log_actual = np.log1p(actual_cases)
    
    def objective(params):
        try:
            pred, _ = simulator.simulate(lambda_v, params)
            log_pred = np.log1p(pred)
            
            # 加权损失：高病例期权重更高
            weights = 1 + np.log1p(actual_cases) / np.max(np.log1p(actual_cases))
            mse = np.mean(weights * (log_actual - log_pred)**2)
            
            # 相关性
            if np.std(pred) > 0:
                corr = np.corrcoef(actual_cases, pred)[0, 1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = -1
            
            # 惩罚异常值
            if np.max(pred) > actual_cases.max() * 10:
                return 1000
            if np.max(pred) < actual_cases.max() * 0.01:
                return 1000
            
            return mse - 0.3 * corr
            
        except:
            return 1000
    
    # 参数: [beta_vh, beta_hv, b, m_ratio, init_I_v_frac, case_scale]
    bounds = [
        (0.4, 0.8),     # beta_vh
        (0.4, 0.8),     # beta_hv
        (0.4, 0.9),     # b
        (2.0, 8.0),     # m_ratio
        (0.001, 0.05),  # init_I_v_frac
        (0.5, 50.0),    # case_scale
    ]
    
    print("  参数优化中...")
    result = differential_evolution(
        objective, bounds,
        maxiter=80,
        seed=42,
        workers=1,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7
    )
    
    return result.x


def create_visualization(results, save_path):
    """创建可视化"""
    fig = plt.figure(figsize=(16, 12))
    
    actual = results['actual']
    predicted = results['predicted']
    R0 = results['R0']
    months = range(len(actual))
    
    # 1. 时间序列对比 (线性尺度)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(months, actual, 'b-', lw=2, label='Actual', marker='o', ms=3)
    ax1.plot(months, predicted, 'r--', lw=2, label='Predicted', marker='s', ms=3)
    ax1.set_xlabel('Month Index (2006-2014)')
    ax1.set_ylabel('Cases')
    ax1.set_title('Monthly Dengue Cases: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 时间序列对比 (对数尺度)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(months, actual + 1, 'b-', lw=2, label='Actual', marker='o', ms=3)
    ax2.semilogy(months, predicted + 1, 'r--', lw=2, label='Predicted', marker='s', ms=3)
    ax2.set_xlabel('Month Index')
    ax2.set_ylabel('Cases (log scale)')
    ax2.set_title('Cases Comparison (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 散点图
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(actual, predicted, alpha=0.6, c='blue', s=40)
    max_val = max(actual.max(), predicted.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
    ax3.set_xlabel('Actual Cases')
    ax3.set_ylabel('Predicted Cases')
    r2 = r2_score(actual, predicted)
    corr = np.corrcoef(actual, predicted)[0, 1]
    ax3.set_title(f'Prediction vs Actual\nR²={r2:.3f}, r={corr:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('symlog', linthresh=10)
    ax3.set_yscale('symlog', linthresh=10)
    
    # 4. R0时间序列
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(months, R0, 'g-', lw=2)
    ax4.axhline(y=1, color='r', ls='--', label='R0=1')
    ax4.fill_between(months, 0, R0, where=np.array(R0)>1, color='red', alpha=0.3)
    ax4.set_xlabel('Month Index')
    ax4.set_ylabel('R0')
    ax4.set_title(f'Basic Reproduction Number\nMean={np.mean(R0):.2f}, Max={np.max(R0):.2f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 年度对比
    ax5 = fig.add_subplot(2, 3, 5)
    years = results['years']
    yearly_actual = results['yearly_actual']
    yearly_pred = results['yearly_pred']
    x = range(len(years))
    width = 0.35
    ax5.bar([i - width/2 for i in x], yearly_actual, width, label='Actual', color='steelblue')
    ax5.bar([i + width/2 for i in x], yearly_pred, width, label='Predicted', color='coral')
    ax5.set_xticks(x)
    ax5.set_xticklabels(years)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Annual Cases')
    ax5.set_title('Annual Cases Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('symlog', linthresh=100)
    
    # 6. 性能指标
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # 对数空间的R²
    log_r2 = r2_score(np.log1p(actual), np.log1p(predicted))
    
    text = f"""
    Two-Stage SEI-SEIR Model Results
    ================================
    
    Data: Guangdong Province
    Period: 2006-2014 (Monthly)
    Samples: {len(actual)} months
    
    Performance Metrics:
    - R² (linear): {r2:.4f}
    - R² (log): {log_r2:.4f}
    - Correlation: {corr:.4f}
    - MAE: {mae:.0f} cases
    - RMSE: {rmse:.0f} cases
    
    R0 Statistics:
    - Mean: {np.mean(R0):.2f}
    - Max: {np.max(R0):.2f}
    - Months R0>1: {(np.array(R0)>1).sum()}
    
    Calibrated Parameters:
    - β_vh: {results['params'][0]:.3f}
    - β_hv: {results['params'][1]:.3f}
    - b: {results['params'][2]:.3f}
    - m_ratio: {results['params'][3]:.2f}
    """
    
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("两阶段SEI-SEIR模型 V2：病例数拟合")
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
    
    # 筛选分析时段
    analysis_df = case_df[(case_df['year'] >= 2006) & (case_df['year'] <= 2014)].copy()
    print(f"  时段: 2006-2014年, {len(analysis_df)}个月")
    print(f"  总病例: {analysis_df['cases'].sum():,}")
    
    # 训练TCN
    print("\n[2] 训练TCN预测蚊虫动态...")
    weather_df = analysis_df[['year', 'month', 'temperature', 'humidity', 'precipitation']].copy()
    tcn_model, scaler_X, scaler_y, seq_len = train_tcn_for_lambda(weather_df, gz_bi)
    
    # 预测λ_v
    print("\n[3] 预测λ_v序列...")
    lambda_v = predict_lambda_v(tcn_model, scaler_X, scaler_y, analysis_df, seq_len)
    print(f"  λ_v范围: [{lambda_v.min():.0f}, {lambda_v.max():.0f}]")
    
    # 校准参数
    print("\n[4] 校准SEI-SEIR参数...")
    actual_cases = analysis_df['cases'].values
    best_params = calibrate_params(lambda_v, actual_cases)
    
    print("\n  校准后参数:")
    param_names = ['beta_vh', 'beta_hv', 'b', 'm_ratio', 'init_I_v_frac', 'case_scale']
    for name, val in zip(param_names, best_params):
        print(f"    {name}: {val:.4f}")
    
    # 生成预测
    print("\n[5] 生成病例预测...")
    simulator = ImprovedSEISEIR()
    predicted, R0 = simulator.simulate(lambda_v, best_params)
    
    # 评估
    print("\n[6] 模型评估...")
    r2 = r2_score(actual_cases, predicted)
    log_r2 = r2_score(np.log1p(actual_cases), np.log1p(predicted))
    corr = np.corrcoef(actual_cases, predicted)[0, 1]
    mae = mean_absolute_error(actual_cases, predicted)
    
    print(f"  R² (线性): {r2:.4f}")
    print(f"  R² (对数): {log_r2:.4f}")
    print(f"  相关系数: {corr:.4f}")
    print(f"  MAE: {mae:.0f}")
    print(f"  R0均值: {np.mean(R0):.2f}, 最大值: {np.max(R0):.2f}")
    
    # 年度统计
    analysis_df['predicted'] = predicted
    yearly_actual = analysis_df.groupby('year')['cases'].sum().values
    yearly_pred = analysis_df.groupby('year')['predicted'].sum().values
    years = analysis_df['year'].unique()
    
    # 可视化
    print("\n[7] 生成可视化...")
    results = {
        'actual': actual_cases,
        'predicted': predicted,
        'R0': R0,
        'years': years,
        'yearly_actual': yearly_actual,
        'yearly_pred': yearly_pred,
        'params': best_params
    }
    
    save_path = '../results/two_stage_model_v2_results.png'
    create_visualization(results, save_path)
    print(f"  已保存: {save_path}")
    
    # 保存结果
    results_df = pd.DataFrame({
        'year': analysis_df['year'].values,
        'month': analysis_df['month'].values,
        'actual_cases': actual_cases,
        'predicted_cases': predicted,
        'R0': R0,
        'lambda_v': lambda_v
    })
    results_df.to_csv('../results/two_stage_v2_predictions.csv', index=False)
    print(f"  已保存: ../results/two_stage_v2_predictions.csv")
    
    print("\n" + "=" * 70)
    print("模型训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
