"""
两阶段SEI-SEIR模型：TCN预测蚊虫动态 + ODE求解器预测病例数

第一阶段: 气象数据 → TCN → 蚊虫出生率 λ_v(t)
第二阶段: λ_v(t) → SEI-SEIR ODE → 月度新增病例

使用广东省实际病例数据进行参数校准
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


class TemporalBlock(nn.Module):
    """TCN时间卷积块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.padding = padding
        
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """TCN模型 - 预测蚊虫出生率"""
    def __init__(self, input_size, hidden_size=64, num_levels=3, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)


class SEISEIRSimulator:
    """SEI-SEIR动力学模拟器"""
    
    def __init__(self, N_h=110000000):
        """
        初始化参数
        N_h: 人口数量 (广东省约1.1亿)
        """
        self.N_h = N_h
        
        # 固定的生物学参数 (来自文献)
        self.mu_v = 1/14       # 蚊虫死亡率 (寿命14天)
        self.sigma_v = 1/10    # 蚊虫潜伏期转化率 (EIP 10天)
        self.sigma_h = 1/5.5   # 人潜伏期转化率 (5.5天)
        self.gamma_h = 1/7     # 人恢复率 (7天)
        
        # 待校准的传播参数
        self.beta_vh = 0.5     # 蚊→人传播概率
        self.beta_hv = 0.5     # 人→蚊传播概率
        self.b = 0.5           # 叮咬率
        self.m_ratio = 2.0     # 蚊人比
        
    def set_transmission_params(self, beta_vh, beta_hv, b, m_ratio):
        """设置传播参数"""
        self.beta_vh = beta_vh
        self.beta_hv = beta_hv
        self.b = b
        self.m_ratio = m_ratio
        
    def simulate_monthly(self, lambda_v_series, I_h_init=10, import_rate=0.0):
        """
        模拟月度病例数
        
        Parameters:
        -----------
        lambda_v_series: 蚊虫出生率序列 (月度)
        I_h_init: 初始感染人数
        import_rate: 输入性病例率
        
        Returns:
        --------
        new_cases: 月度新增病例数组
        """
        months = len(lambda_v_series)
        
        # 初始条件
        N_v = self.N_h * self.m_ratio
        S_v = N_v * 0.95
        E_v = N_v * 0.03
        I_v = N_v * 0.02
        
        S_h = self.N_h - I_h_init
        E_h = 0
        I_h = I_h_init
        R_h = 0
        
        new_cases_monthly = []
        R0_series = []
        I_h_series = []
        
        dt = 1  # 1天步长
        days_per_month = 30
        
        for m in range(months):
            lambda_v = lambda_v_series[m]
            monthly_new_infections = 0
            
            for d in range(days_per_month):
                # 蚊虫动态
                force_vh = self.b * self.beta_hv * I_h / self.N_h
                dS_v = lambda_v - force_vh * S_v - self.mu_v * S_v
                dE_v = force_vh * S_v - self.sigma_v * E_v - self.mu_v * E_v
                dI_v = self.sigma_v * E_v - self.mu_v * I_v
                
                # 人群动态
                force_hv = self.b * self.beta_vh * I_v / self.N_h
                new_exposed = force_hv * S_h
                new_infected = self.sigma_h * E_h
                
                # 输入性病例
                imported = import_rate * self.N_h / 365 / 30
                
                dS_h = -force_hv * S_h - imported
                dE_h = force_hv * S_h + imported - self.sigma_h * E_h
                dI_h = self.sigma_h * E_h - self.gamma_h * I_h
                dR_h = self.gamma_h * I_h
                
                # 更新状态
                S_v = max(0, S_v + dS_v * dt)
                E_v = max(0, E_v + dE_v * dt)
                I_v = max(0, I_v + dI_v * dt)
                S_h = max(0, S_h + dS_h * dt)
                E_h = max(0, E_h + dE_h * dt)
                I_h = max(0, I_h + dI_h * dt)
                R_h = max(0, R_h + dR_h * dt)
                
                monthly_new_infections += new_infected
            
            new_cases_monthly.append(monthly_new_infections)
            
            # 计算R0
            N_v_current = S_v + E_v + I_v
            m = N_v_current / self.N_h
            R0 = (self.b ** 2 * self.beta_vh * self.beta_hv * m * self.sigma_v) / \
                 ((self.sigma_v + self.mu_v) * self.mu_v * self.gamma_h)
            R0_series.append(R0)
            I_h_series.append(I_h)
        
        return np.array(new_cases_monthly), np.array(R0_series), np.array(I_h_series)


class TwoStageModel:
    """两阶段模型：TCN + SEI-SEIR"""
    
    def __init__(self, N_h=110000000):
        self.N_h = N_h
        self.tcn = None
        self.simulator = SEISEIRSimulator(N_h)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.seq_length = 6  # 使用6个月历史数据
        
    def prepare_tcn_data(self, weather_df, bi_df):
        """准备TCN训练数据"""
        # 合并数据
        merged = pd.merge(weather_df, bi_df, on=['year', 'month'], how='left')
        merged = merged.dropna(subset=['bi'])
        
        # 特征: 温度、湿度、降水
        features = ['temperature', 'humidity', 'precipitation']
        X = merged[features].values
        y = merged['bi'].values.reshape(-1, 1)
        
        # 归一化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 创建序列
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.seq_length:i])
            y_seq.append(y_scaled[i])
        
        return np.array(X_seq), np.array(y_seq), merged
    
    def train_tcn(self, X, y, epochs=200, lr=0.001):
        """训练TCN模型"""
        # 转换为tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # 划分训练集和验证集
        split = int(0.8 * len(X))
        X_train, X_val = X_tensor[:split], X_tensor[split:]
        y_train, y_val = y_tensor[:split], y_tensor[split:]
        
        # 初始化模型
        self.tcn = TCNModel(input_size=X.shape[2], hidden_size=32, num_levels=3)
        optimizer = torch.optim.Adam(self.tcn.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 训练
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            self.tcn.train()
            optimizer.zero_grad()
            pred = self.tcn(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # 验证
            self.tcn.eval()
            with torch.no_grad():
                val_pred = self.tcn(X_val)
                val_loss = criterion(val_pred, y_val)
                val_losses.append(val_loss.item())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.tcn.state_dict().copy()
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        self.tcn.load_state_dict(best_state)
        return train_losses, val_losses
    
    def predict_lambda_v(self, weather_data):
        """使用TCN预测蚊虫出生率"""
        self.tcn.eval()
        
        features = ['temperature', 'humidity', 'precipitation']
        X = weather_data[features].values
        X_scaled = self.scaler_X.transform(X)
        
        lambda_v_series = []
        
        for i in range(len(X_scaled)):
            if i < self.seq_length:
                # 前几个月使用简单公式
                temp, humid, precip = X[i]
                lambda_v = self._compute_lambda_v_simple(temp, humid, precip)
            else:
                # 使用TCN预测
                seq = X_scaled[i-self.seq_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                with torch.no_grad():
                    bi_pred = self.tcn(seq_tensor).item()
                bi_pred = self.scaler_y.inverse_transform([[bi_pred]])[0, 0]
                # BI转换为λ_v (简化关系)
                lambda_v = max(100, bi_pred * 500)  # 放大系数
            
            lambda_v_series.append(lambda_v)
        
        return np.array(lambda_v_series)
    
    def _compute_lambda_v_simple(self, temp, humid, precip):
        """简单的λ_v计算公式"""
        T_opt, T_width = 26.5, 8
        H_opt, H_width = 75, 20
        
        temp_factor = np.exp(-((temp - T_opt) / T_width) ** 2)
        humid_factor = np.exp(-((humid - H_opt) / H_width) ** 2)
        precip_factor = np.tanh(precip / 5)
        
        return 5000 * temp_factor * humid_factor * (0.3 + 0.7 * precip_factor)
    
    def calibrate_seir_params(self, lambda_v_series, actual_cases, 
                               initial_infected=10, max_iter=100):
        """
        使用实际病例数据校准SEI-SEIR参数
        
        Parameters:
        -----------
        lambda_v_series: TCN预测的蚊虫出生率
        actual_cases: 实际月度病例数
        """
        print("\n[参数校准] 使用差分进化算法优化传播参数...")
        
        def objective(params):
            beta_vh, beta_hv, b, m_ratio, import_rate, scale = params
            
            self.simulator.set_transmission_params(beta_vh, beta_hv, b, m_ratio)
            
            try:
                pred_cases, _, _ = self.simulator.simulate_monthly(
                    lambda_v_series, 
                    I_h_init=initial_infected,
                    import_rate=import_rate
                )
                
                # 应用缩放因子
                pred_cases = pred_cases * scale
                
                # 计算损失 (对数空间，避免大值主导)
                actual_log = np.log1p(actual_cases)
                pred_log = np.log1p(pred_cases)
                
                mse = np.mean((actual_log - pred_log) ** 2)
                
                # 相关性奖励
                if np.std(pred_cases) > 0:
                    corr = np.corrcoef(actual_cases, pred_cases)[0, 1]
                    if np.isnan(corr):
                        corr = 0
                else:
                    corr = 0
                
                return mse - 0.5 * corr  # 最小化MSE，最大化相关性
                
            except Exception as e:
                return 1e10
        
        # 参数边界
        bounds = [
            (0.3, 0.9),      # beta_vh
            (0.3, 0.9),      # beta_hv
            (0.3, 1.0),      # b (叮咬率)
            (1.0, 10.0),     # m_ratio (蚊人比)
            (1e-8, 1e-5),    # import_rate (输入率)
            (0.1, 100.0),    # scale (缩放因子)
        ]
        
        result = differential_evolution(
            objective, bounds, 
            maxiter=max_iter, 
            seed=42,
            workers=1,  # 单进程避免pickle问题
            disp=True,
            polish=True
        )
        
        best_params = result.x
        param_names = ['beta_vh', 'beta_hv', 'b', 'm_ratio', 'import_rate', 'scale']
        
        print("\n校准后的参数:")
        for name, val in zip(param_names, best_params):
            print(f"  {name}: {val:.6f}")
        
        # 设置最优参数
        self.simulator.set_transmission_params(*best_params[:4])
        self.calibrated_params = dict(zip(param_names, best_params))
        
        return best_params
    
    def predict_cases(self, lambda_v_series, initial_infected=10):
        """预测病例数"""
        pred_cases, R0_series, I_h_series = self.simulator.simulate_monthly(
            lambda_v_series,
            I_h_init=initial_infected,
            import_rate=self.calibrated_params.get('import_rate', 0)
        )
        
        # 应用缩放
        scale = self.calibrated_params.get('scale', 1.0)
        pred_cases = pred_cases * scale
        
        return pred_cases, R0_series, I_h_series


def load_data():
    """加载所有数据"""
    # 病例数据
    case_df = pd.read_csv('../data/guangdong_dengue_cases.csv')
    case_df['date'] = pd.to_datetime(case_df['date'])
    
    # BI数据
    bi_df = pd.read_csv('../data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    
    return case_df, gz_bi


def create_visualization(results, save_path):
    """创建综合可视化"""
    fig = plt.figure(figsize=(16, 14))
    
    months = results['months']
    actual = results['actual_cases']
    predicted = results['predicted_cases']
    R0 = results['R0']
    
    # 1. 病例拟合对比
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(months, actual, 'b-', linewidth=2, label='Actual Cases', marker='o', markersize=4)
    ax1.plot(months, predicted, 'r--', linewidth=2, label='Predicted Cases', marker='s', markersize=4)
    ax1.fill_between(months, 0, actual, alpha=0.3, color='blue')
    ax1.set_xlabel('Month Index')
    ax1.set_ylabel('Cases')
    ax1.set_title('Dengue Cases: Actual vs Predicted (Two-Stage Model)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog', linthresh=10)  # 对数尺度便于观察
    
    # 2. 预测vs实际散点图
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.scatter(actual, predicted, alpha=0.6, c='blue', s=40)
    max_val = max(actual.max(), predicted.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Cases')
    ax2.set_ylabel('Predicted Cases')
    r2 = r2_score(actual, predicted)
    corr = np.corrcoef(actual, predicted)[0, 1]
    ax2.set_title(f'Prediction vs Actual\nR²={r2:.3f}, Correlation={corr:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('symlog', linthresh=10)
    ax2.set_yscale('symlog', linthresh=10)
    
    # 3. R0时间序列
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(months, R0, 'g-', linewidth=2)
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='R0=1 (Epidemic Threshold)')
    ax3.fill_between(months, 0, R0, where=np.array(R0)>1, color='red', alpha=0.3, label='Epidemic Risk')
    ax3.set_xlabel('Month Index')
    ax3.set_ylabel('R0')
    ax3.set_title(f'Basic Reproduction Number R0\nMean={np.mean(R0):.2f}, Max={np.max(R0):.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 年度对比
    ax4 = fig.add_subplot(3, 2, 4)
    years = results['years']
    actual_yearly = results['actual_yearly']
    pred_yearly = results['pred_yearly']
    x = range(len(years))
    width = 0.35
    ax4.bar([i - width/2 for i in x], actual_yearly, width, label='Actual', color='steelblue', alpha=0.7)
    ax4.bar([i + width/2 for i in x], pred_yearly, width, label='Predicted', color='coral', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(years)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Annual Cases')
    ax4.set_title('Annual Cases: Actual vs Predicted')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('symlog', linthresh=100)
    
    # 5. 2014年暴发详情
    ax5 = fig.add_subplot(3, 2, 5)
    idx_2014 = results['idx_2014']
    if len(idx_2014) > 0:
        months_2014 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(idx_2014)]
        actual_2014 = actual[idx_2014]
        pred_2014 = predicted[idx_2014]
        x_2014 = range(len(idx_2014))
        ax5.bar([i - 0.2 for i in x_2014], actual_2014, 0.4, label='Actual', color='red', alpha=0.7)
        ax5.bar([i + 0.2 for i in x_2014], pred_2014, 0.4, label='Predicted', color='orange', alpha=0.7)
        ax5.set_xticks(x_2014)
        ax5.set_xticklabels(months_2014)
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Cases')
        ax5.set_title('2014 Outbreak: Monthly Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('symlog', linthresh=100)
    
    # 6. 模型性能总结
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    
    # 计算评估指标
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
    
    summary_text = f"""
    Two-Stage SEI-SEIR Model Performance
    =====================================
    
    Data: Guangdong Province, 2006-2014
    Time Scale: Monthly
    Total Months: {len(actual)}
    
    Model Metrics:
    - R²: {r2:.4f}
    - Correlation: {corr:.4f}
    - MAE: {mae:.1f} cases
    - RMSE: {rmse:.1f} cases
    - MAPE: {mape:.1f}%
    
    R0 Statistics:
    - Mean R0: {np.mean(R0):.3f}
    - Max R0: {np.max(R0):.3f}
    - Months with R0>1: {(np.array(R0)>1).sum()}
    
    Calibrated Parameters:
    - β_vh: {results['params']['beta_vh']:.4f}
    - β_hv: {results['params']['beta_hv']:.4f}
    - b: {results['params']['b']:.4f}
    - m (mosquito ratio): {results['params']['m_ratio']:.4f}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


def main():
    print("=" * 70)
    print("两阶段SEI-SEIR模型：病例数拟合")
    print("第一阶段: TCN预测蚊虫动态 | 第二阶段: SEI-SEIR预测病例")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    case_df, bi_df = load_data()
    
    # 筛选2006-2014年 (有BI数据的时段)
    analysis_df = case_df[(case_df['year'] >= 2006) & (case_df['year'] <= 2014)].copy()
    print(f"  分析时段: 2006-2014年")
    print(f"  总月份数: {len(analysis_df)}")
    print(f"  总病例数: {analysis_df['cases'].sum():,}")
    
    # 2. 初始化两阶段模型
    print("\n[2] 初始化两阶段模型...")
    model = TwoStageModel(N_h=110000000)
    
    # 3. 准备TCN数据并训练
    print("\n[3] 训练TCN模型 (预测蚊虫动态)...")
    weather_df = analysis_df[['year', 'month', 'temperature', 'humidity', 'precipitation']].copy()
    X, y, merged_df = model.prepare_tcn_data(weather_df, bi_df)
    
    if len(X) > 0:
        train_losses, val_losses = model.train_tcn(X, y, epochs=200)
        print(f"  TCN训练完成，最终验证损失: {val_losses[-1]:.4f}")
    
    # 4. 预测λ_v序列
    print("\n[4] 预测蚊虫出生率 λ_v(t)...")
    lambda_v_series = model.predict_lambda_v(analysis_df)
    print(f"  λ_v 范围: [{lambda_v_series.min():.1f}, {lambda_v_series.max():.1f}]")
    
    # 5. 校准SEI-SEIR参数
    print("\n[5] 校准SEI-SEIR传播参数...")
    actual_cases = analysis_df['cases'].values
    best_params = model.calibrate_seir_params(
        lambda_v_series, 
        actual_cases,
        initial_infected=actual_cases[0] if actual_cases[0] > 0 else 10,
        max_iter=50
    )
    
    # 6. 预测病例数
    print("\n[6] 生成病例预测...")
    predicted_cases, R0_series, I_h_series = model.predict_cases(
        lambda_v_series,
        initial_infected=actual_cases[0] if actual_cases[0] > 0 else 10
    )
    
    # 7. 评估结果
    print("\n[7] 模型评估...")
    r2 = r2_score(actual_cases, predicted_cases)
    corr = np.corrcoef(actual_cases, predicted_cases)[0, 1]
    mae = mean_absolute_error(actual_cases, predicted_cases)
    rmse = np.sqrt(mean_squared_error(actual_cases, predicted_cases))
    
    print(f"  R²: {r2:.4f}")
    print(f"  相关系数: {corr:.4f}")
    print(f"  MAE: {mae:.1f} cases")
    print(f"  RMSE: {rmse:.1f} cases")
    print(f"  R0均值: {np.mean(R0_series):.3f}")
    print(f"  R0最大值: {np.max(R0_series):.3f}")
    
    # 8. 年度统计
    analysis_df['predicted'] = predicted_cases
    yearly_actual = analysis_df.groupby('year')['cases'].sum().values
    yearly_pred = analysis_df.groupby('year')['predicted'].sum().values
    years = analysis_df['year'].unique()
    
    # 找到2014年的索引
    idx_2014 = np.where(analysis_df['year'] == 2014)[0]
    
    # 9. 可视化
    print("\n[8] 生成可视化...")
    results = {
        'months': range(len(actual_cases)),
        'actual_cases': actual_cases,
        'predicted_cases': predicted_cases,
        'R0': R0_series,
        'years': years,
        'actual_yearly': yearly_actual,
        'pred_yearly': yearly_pred,
        'idx_2014': idx_2014,
        'params': model.calibrated_params
    }
    
    save_path = '../results/two_stage_model_results.png'
    create_visualization(results, save_path)
    print(f"  已保存: {save_path}")
    
    # 10. 保存预测结果
    results_df = pd.DataFrame({
        'year': analysis_df['year'].values,
        'month': analysis_df['month'].values,
        'actual_cases': actual_cases,
        'predicted_cases': predicted_cases,
        'R0': R0_series,
        'lambda_v': lambda_v_series
    })
    results_df.to_csv('../results/two_stage_predictions.csv', index=False)
    print(f"  已保存: ../results/two_stage_predictions.csv")
    
    print("\n" + "=" * 70)
    print("两阶段模型训练完成!")
    print("=" * 70)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
