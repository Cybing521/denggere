"""
SEI-SEIR 登革热传播动力学模型 (周数据版本)

模型结构:
- 蚊虫: S_v → E_v → I_v (SEI)
- 人群: S_h → E_h → I_h → R_h (SEIR)

参数设定:
- 生物学参数: 来自文献，固定
- Λ_v(t): 蚊虫出生率，用TCN从气象数据预测

时间尺度: 周 (Weekly)
数据范围: 2015-2019年
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. SEI-SEIR 模型参数 (来自文献，固定)
# ============================================================
class ModelParameters:
    """
    SEI-SEIR模型参数 (表4)
    所有参数来自文献，固定不变
    """
    # 蚊虫参数
    mu_v = 0.05          # 蚊虫死亡率 (1/天), 约20天寿命
    beta_v = 0.5         # 人→蚊传播概率
    beta_h = 0.75        # 蚊→人传播概率  
    b = 0.5              # 叮咬率 (次/天)
    sigma_v = 0.1        # 蚊虫潜伏期转化率 (1/天), EIP≈10天
    
    # 人群参数
    sigma_h = 0.2        # 人潜伏期转化率 (1/天), 约5天
    gamma = 0.143        # 康复率 (1/天), 约7天
    
    # 人口
    N_h = 14_000_000     # 广州人口
    
    @classmethod
    def print_params(cls):
        """打印参数"""
        print("SEI-SEIR 模型参数 (文献值):")
        print(f"  蚊虫死亡率 μ_v = {cls.mu_v} /天 (寿命≈{1/cls.mu_v:.0f}天)")
        print(f"  人→蚊传播概率 β_v = {cls.beta_v}")
        print(f"  蚊→人传播概率 β_h = {cls.beta_h}")
        print(f"  叮咬率 b = {cls.b} /天")
        print(f"  蚊虫潜伏期转化率 σ_v = {cls.sigma_v} /天 (EIP≈{1/cls.sigma_v:.0f}天)")
        print(f"  人潜伏期转化率 σ_h = {cls.sigma_h} /天 (潜伏期≈{1/cls.sigma_h:.0f}天)")
        print(f"  康复率 γ = {cls.gamma} /天 (感染期≈{1/cls.gamma:.0f}天)")
        print(f"  人口 N_h = {cls.N_h:,}")


# ============================================================
# 2. SEI-SEIR 微分方程组
# ============================================================
def sei_seir_equations(t, y, Lambda_v_func, params):
    """
    SEI-SEIR 微分方程组
    
    蚊虫 (SEI):
        dS_v/dt = Λ_v(t) - b·β_v·S_v·I_h/N_h - μ_v·S_v
        dE_v/dt = b·β_v·S_v·I_h/N_h - σ_v·E_v - μ_v·E_v
        dI_v/dt = σ_v·E_v - μ_v·I_v
    
    人群 (SEIR):
        dS_h/dt = -b·β_h·S_h·I_v/N_h
        dE_h/dt = b·β_h·S_h·I_v/N_h - σ_h·E_h
        dI_h/dt = σ_h·E_h - γ·I_h
        dR_h/dt = γ·I_h
    
    Args:
        t: 时间 (天)
        y: [S_v, E_v, I_v, S_h, E_h, I_h, R_h]
        Lambda_v_func: Λ_v(t) 插值函数
        params: ModelParameters
    """
    S_v, E_v, I_v, S_h, E_h, I_h, R_h = y
    
    # 参数
    mu_v = params.mu_v
    beta_v = params.beta_v
    beta_h = params.beta_h
    b = params.b
    sigma_v = params.sigma_v
    sigma_h = params.sigma_h
    gamma = params.gamma
    N_h = params.N_h
    
    # 蚊虫出生率 (时变，从TCN预测)
    Lambda_v = Lambda_v_func(t)
    
    # 蚊虫总数
    N_v = S_v + E_v + I_v
    
    # 蚊虫 SEI
    dS_v = Lambda_v - b * beta_v * S_v * I_h / N_h - mu_v * S_v
    dE_v = b * beta_v * S_v * I_h / N_h - sigma_v * E_v - mu_v * E_v
    dI_v = sigma_v * E_v - mu_v * I_v
    
    # 人群 SEIR
    dS_h = -b * beta_h * S_h * I_v / N_h
    dE_h = b * beta_h * S_h * I_v / N_h - sigma_h * E_h
    dI_h = sigma_h * E_h - gamma * I_h
    dR_h = gamma * I_h
    
    return [dS_v, dE_v, dI_v, dS_h, dE_h, dI_h, dR_h]


def solve_sei_seir(y0, t_span, t_eval, Lambda_v_func, params):
    """求解SEI-SEIR方程组"""
    sol = solve_ivp(
        sei_seir_equations,
        t_span,
        y0,
        args=(Lambda_v_func, params),
        t_eval=t_eval,
        method='RK45',
        max_step=1.0
    )
    return sol


def compute_R0(Lambda_v, params):
    """
    计算基本再生数 R0
    
    R0 = (b² · β_v · β_h · σ_v · Λ_v) / (μ_v · γ · N_h · (σ_v + μ_v) · μ_v)
    
    简化: R0 ≈ (b² · β_v · β_h · N_v) / (μ_v · γ · N_h)
    其中 N_v ≈ Λ_v / μ_v (平衡态蚊虫数)
    """
    b = params.b
    beta_v = params.beta_v
    beta_h = params.beta_h
    mu_v = params.mu_v
    sigma_v = params.sigma_v
    gamma = params.gamma
    N_h = params.N_h
    
    # 平衡态蚊虫数
    N_v_eq = Lambda_v / mu_v
    
    # R0 公式 (Ross-Macdonald 类型)
    R0 = (b**2 * beta_v * beta_h * N_v_eq) / (mu_v * gamma * N_h)
    
    return R0


# ============================================================
# 3. TCN 模型 (预测 Λ_v)
# ============================================================
class TCNBlock(nn.Module):
    """TCN基本块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # 因果卷积裁剪
        out = self.relu(out)
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """TCN预测蚊虫出生率"""
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size=3, dilation=dilation))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        out = self.fc(out)
        return out.squeeze(-1)


# ============================================================
# 4. 数据准备 (周数据)
# ============================================================
def prepare_weekly_data():
    """
    准备周数据
    将月数据插值到周分辨率
    """
    print("\n[1] 准备周数据...")
    
    # 加载月数据
    case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
    case_df['date'] = pd.to_datetime(case_df['date'])
    
    # BI数据
    bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    
    # 2015-2019年数据
    df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()
    df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
    df['bi'] = df['bi'].fillna(df['bi'].mean())
    
    # 如果BI缺失太多，用温度估计
    if df['bi'].isna().sum() > len(df) * 0.3:
        df['bi'] = np.exp(-((df['temperature'] - 27) / 8) ** 2) * 5
    
    print(f"  月数据: {len(df)} 个月")
    print(f"  时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"  总病例: {df['cases'].sum():,}")
    
    # 创建周数据
    # 每月约4.33周
    n_months = len(df)
    n_weeks = int(n_months * 4.33)
    
    # 月数据的时间点 (周为单位)
    monthly_weeks = np.arange(n_months) * 4.33
    
    # 周数据的时间点
    weekly_t = np.arange(n_weeks)
    
    # 插值函数
    temp_interp = interp1d(monthly_weeks, df['temperature'].values, kind='linear', fill_value='extrapolate')
    humid_interp = interp1d(monthly_weeks, df['humidity'].values, kind='linear', fill_value='extrapolate')
    precip_interp = interp1d(monthly_weeks, df['precipitation'].values, kind='linear', fill_value='extrapolate')
    bi_interp = interp1d(monthly_weeks, df['bi'].values, kind='linear', fill_value='extrapolate')
    cases_interp = interp1d(monthly_weeks, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')  # 周病例
    
    # 生成周数据
    weekly_data = pd.DataFrame({
        'week': weekly_t,
        'temperature': temp_interp(weekly_t),
        'humidity': humid_interp(weekly_t),
        'precipitation': precip_interp(weekly_t),
        'bi': bi_interp(weekly_t),
        'cases': np.maximum(0, cases_interp(weekly_t))  # 周病例数
    })
    
    # 添加年月信息
    weekly_data['year'] = 2015 + (weekly_data['week'] // 52).astype(int)
    weekly_data['week_of_year'] = (weekly_data['week'] % 52) + 1
    
    print(f"  周数据: {len(weekly_data)} 周")
    print(f"  周均病例: {weekly_data['cases'].mean():.1f}")
    
    return weekly_data, df


# ============================================================
# 5. 训练TCN预测Λ_v
# ============================================================
def train_tcn_for_lambda_v(weekly_data, seq_len=8):
    """
    训练TCN预测蚊虫出生率Λ_v
    
    思路: 用气象数据预测BI，BI与Λ_v成正比
    Λ_v(t) = k × BI(t) × N_h  (k为比例系数)
    """
    print("\n[2] 训练TCN预测Λ_v...")
    
    # 特征: 温度、湿度、降水
    features = ['temperature', 'humidity', 'precipitation']
    X_raw = weekly_data[features].values
    y_raw = weekly_data['bi'].values
    
    # 标准化
    X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
    y_mean, y_std = y_raw.mean(), y_raw.std()
    
    X_scaled = (X_raw - X_mean) / (X_std + 1e-8)
    y_scaled = (y_raw - y_mean) / (y_std + 1e-8)
    
    # 创建序列数据
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_len:i])
        y_seq.append(y_scaled[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 转换为tensor
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq)
    
    # 划分训练/验证
    split = int(len(X_tensor) * 0.8)
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    y_train, y_val = y_tensor[:split], y_tensor[split:]
    
    # 模型
    model = TCNModel(input_size=len(features), hidden_size=32, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(200):
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
            val_loss = criterion(val_pred, y_val)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_state)
    
    # 预测全量
    model.eval()
    with torch.no_grad():
        all_pred_scaled = model(X_tensor).numpy()
    
    # 反标准化
    bi_pred = all_pred_scaled * y_std + y_mean
    bi_actual = y_raw[seq_len:]
    
    # 评估
    corr, _ = pearsonr(bi_actual, bi_pred)
    r2 = r2_score(bi_actual, bi_pred)
    
    print(f"  TCN预测BI性能:")
    print(f"    相关系数: {corr:.4f}")
    print(f"    R²: {r2:.4f}")
    
    # 计算Λ_v
    # Λ_v 与 BI 成正比，需要校准比例系数
    # 假设平衡态: N_v = Λ_v / μ_v, 且 BI 反映 N_v
    # 初始估计: Λ_v = μ_v × k × BI × N_h / 1000
    
    k_scale = 1e6  # 比例系数，待后续校准
    Lambda_v_pred = ModelParameters.mu_v * k_scale * np.maximum(bi_pred, 0.1)
    
    return {
        'model': model,
        'Lambda_v': Lambda_v_pred,
        'bi_pred': bi_pred,
        'bi_actual': bi_actual,
        'offset': seq_len,
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'k_scale': k_scale,
        'corr': corr,
        'r2': r2
    }


# ============================================================
# 6. 校准Λ_v比例系数
# ============================================================
def calibrate_lambda_scale(weekly_data, tcn_result, params):
    """
    校准Λ_v的比例系数k
    使模型输出与观测病例匹配
    """
    print("\n[3] 校准Λ_v比例系数...")
    
    offset = tcn_result['offset']
    bi_pred = tcn_result['bi_pred']
    observed_cases = weekly_data['cases'].values[offset:]
    
    # 时间点 (天)
    n_weeks = len(bi_pred)
    time_days = np.arange(n_weeks) * 7  # 周转天
    
    def objective(params_opt):
        """目标函数"""
        k_scale, I0_frac = params_opt
        
        # Λ_v(t)
        Lambda_v = ModelParameters.mu_v * k_scale * np.maximum(bi_pred, 0.1)
        Lambda_v_func = interp1d(time_days, Lambda_v, kind='linear', fill_value='extrapolate')
        
        # 初始条件
        N_h = params.N_h
        I_h0 = max(1, N_h * I0_frac)
        E_h0 = I_h0 * 2
        S_h0 = N_h - E_h0 - I_h0
        R_h0 = 0
        
        # 蚊虫初始 (平衡态)
        Lambda_v_0 = Lambda_v[0]
        N_v0 = Lambda_v_0 / params.mu_v
        I_v0 = N_v0 * 0.01  # 1%感染
        E_v0 = N_v0 * 0.02
        S_v0 = N_v0 - E_v0 - I_v0
        
        y0 = [S_v0, E_v0, I_v0, S_h0, E_h0, I_h0, R_h0]
        
        try:
            sol = solve_sei_seir(y0, (0, time_days[-1]), time_days, Lambda_v_func, params)
            
            if sol.status != 0:
                return 1e10
            
            # 周新增病例 = γ × I_h × 7 (一周的累积)
            I_h = sol.y[5]
            weekly_new = params.gamma * I_h * 7
            
            # 对数空间MSE
            obs_log = np.log1p(observed_cases)
            pred_log = np.log1p(weekly_new)
            
            mse = np.mean((obs_log - pred_log) ** 2)
            
            # 相关性惩罚
            if len(observed_cases) > 10:
                corr, _ = pearsonr(observed_cases, weekly_new)
                if corr < 0:
                    mse += 5 * (1 - corr)
            
            return mse
            
        except Exception:
            return 1e10
    
    # 优化
    bounds = [
        (1e4, 1e8),    # k_scale
        (1e-9, 1e-5),  # I0_frac
    ]
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        workers=1
    )
    
    k_scale_opt, I0_frac_opt = result.x
    print(f"  优化结果:")
    print(f"    k_scale = {k_scale_opt:.2e}")
    print(f"    I0 = {params.N_h * I0_frac_opt:.1f}")
    print(f"    Loss = {result.fun:.4f}")
    
    return k_scale_opt, I0_frac_opt


# ============================================================
# 7. 完整模型运行
# ============================================================
def run_sei_seir_model():
    """运行完整的SEI-SEIR模型"""
    
    print("=" * 70)
    print("SEI-SEIR 登革热传播动力学模型")
    print("时间尺度: 周 (Weekly)")
    print("数据范围: 2015-2019年")
    print("=" * 70)
    
    # 打印参数
    params = ModelParameters()
    params.print_params()
    
    # 准备数据
    weekly_data, monthly_data = prepare_weekly_data()
    
    # 训练TCN
    tcn_result = train_tcn_for_lambda_v(weekly_data)
    
    # 校准比例系数
    k_scale, I0_frac = calibrate_lambda_scale(weekly_data, tcn_result, params)
    
    # 运行最终模型
    print("\n[4] 运行SEI-SEIR模型...")
    
    offset = tcn_result['offset']
    bi_pred = tcn_result['bi_pred']
    observed_cases = weekly_data['cases'].values[offset:]
    
    n_weeks = len(bi_pred)
    time_days = np.arange(n_weeks) * 7
    
    # Λ_v(t)
    Lambda_v = params.mu_v * k_scale * np.maximum(bi_pred, 0.1)
    Lambda_v_func = interp1d(time_days, Lambda_v, kind='linear', fill_value='extrapolate')
    
    # 初始条件
    N_h = params.N_h
    I_h0 = max(1, N_h * I0_frac)
    E_h0 = I_h0 * 2
    S_h0 = N_h - E_h0 - I_h0
    R_h0 = 0
    
    Lambda_v_0 = Lambda_v[0]
    N_v0 = Lambda_v_0 / params.mu_v
    I_v0 = N_v0 * 0.01
    E_v0 = N_v0 * 0.02
    S_v0 = N_v0 - E_v0 - I_v0
    
    y0 = [S_v0, E_v0, I_v0, S_h0, E_h0, I_h0, R_h0]
    
    # 求解
    sol = solve_sei_seir(y0, (0, time_days[-1]), time_days, Lambda_v_func, params)
    
    # 结果
    S_v, E_v, I_v = sol.y[0], sol.y[1], sol.y[2]
    S_h, E_h, I_h, R_h = sol.y[3], sol.y[4], sol.y[5], sol.y[6]
    N_v = S_v + E_v + I_v
    
    # 周新增病例
    weekly_new = params.gamma * I_h * 7
    
    # R0(t)
    R0_t = np.array([compute_R0(Lambda_v_func(t), params) for t in time_days])
    
    # 评估
    print("\n[5] 模型评估...")
    
    corr, pval = pearsonr(observed_cases, weekly_new)
    r2_log = r2_score(np.log1p(observed_cases), np.log1p(weekly_new))
    r2_linear = r2_score(observed_cases, weekly_new)
    
    # 趋势准确率
    obs_trend = np.diff(observed_cases) > 0
    pred_trend = np.diff(weekly_new) > 0
    trend_acc = np.mean(obs_trend == pred_trend)
    
    print(f"  病例预测:")
    print(f"    相关系数: {corr:.4f} (p={pval:.6f})")
    print(f"    R² (对数): {r2_log:.4f}")
    print(f"    R² (线性): {r2_linear:.4f}")
    print(f"    趋势准确率: {trend_acc:.2%}")
    
    print(f"\n  R0(t) 统计:")
    print(f"    范围: [{R0_t.min():.4f}, {R0_t.max():.4f}]")
    print(f"    均值: {R0_t.mean():.4f}")
    print(f"    R0>1 周数: {(R0_t > 1).sum()}/{len(R0_t)}")
    
    print(f"\n  蚊虫动态:")
    print(f"    N_v 范围: [{N_v.min():.2e}, {N_v.max():.2e}]")
    print(f"    I_v/N_v 范围: [{(I_v/N_v).min():.4f}, {(I_v/N_v).max():.4f}]")
    
    # 可视化
    print("\n[6] 生成可视化...")
    
    fig = plt.figure(figsize=(18, 16))
    
    weeks = range(len(observed_cases))
    
    # 1. 病例对比
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(weeks, observed_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
    ax1.plot(weeks, weekly_new, 'r-', lw=1.5, label='SEI-SEIR', alpha=0.8)
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Cases')
    ax1.set_title(f'Weekly Cases (r={corr:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 对数尺度
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.semilogy(weeks, observed_cases + 1, 'b-', lw=1.5, label='Observed')
    ax2.semilogy(weeks, weekly_new + 1, 'r-', lw=1.5, label='SEI-SEIR')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Cases (log)')
    ax2.set_title(f'Cases Log Scale (R²={r2_log:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. R0(t)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(weeks, R0_t, 'g-', lw=1.5)
    ax3.axhline(y=1, color='red', ls='--', lw=2, label='R0=1')
    ax3.fill_between(weeks, 0, R0_t, where=R0_t > 1, alpha=0.3, color='red')
    ax3.set_xlabel('Week')
    ax3.set_ylabel('R0(t)')
    ax3.set_title('Basic Reproduction Number R0(t)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Λ_v(t)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(weeks, Lambda_v, 'm-', lw=1.5)
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Λ_v(t)')
    ax4.set_title('Mosquito Birth Rate Λ_v(t)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 蚊虫动态
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(weeks, S_v, 'g-', lw=1, label='S_v', alpha=0.7)
    ax5.plot(weeks, E_v, 'y-', lw=1, label='E_v', alpha=0.7)
    ax5.plot(weeks, I_v, 'r-', lw=1, label='I_v', alpha=0.7)
    ax5.set_xlabel('Week')
    ax5.set_ylabel('Mosquito Population')
    ax5.set_title('Mosquito SEI Dynamics')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 人群动态
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(weeks, E_h, 'y-', lw=1.5, label='E_h', alpha=0.7)
    ax6.plot(weeks, I_h, 'r-', lw=1.5, label='I_h', alpha=0.7)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(weeks, R_h, 'g-', lw=1.5, label='R_h', alpha=0.7)
    ax6.set_xlabel('Week')
    ax6.set_ylabel('E_h, I_h')
    ax6_twin.set_ylabel('R_h (cumulative)')
    ax6.set_title('Human SEIR Dynamics')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # 7. BI预测 vs 实际
    ax7 = fig.add_subplot(3, 3, 7)
    bi_actual = tcn_result['bi_actual']
    bi_pred_plot = tcn_result['bi_pred']
    ax7.plot(weeks, bi_actual, 'b-', lw=1.5, label='Actual BI', alpha=0.8)
    ax7.plot(weeks, bi_pred_plot, 'r-', lw=1.5, label='TCN Predicted', alpha=0.8)
    ax7.set_xlabel('Week')
    ax7.set_ylabel('Breteau Index')
    ax7.set_title(f'BI Prediction (r={tcn_result["corr"]:.3f})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 散点图
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.scatter(observed_cases, weekly_new, c=weeks, cmap='viridis', s=20, alpha=0.6)
    max_val = max(observed_cases.max(), weekly_new.max())
    ax8.plot([0, max_val], [0, max_val], 'k--', lw=2)
    ax8.set_xlabel('Observed Cases')
    ax8.set_ylabel('Predicted Cases')
    ax8.set_title('Observed vs Predicted')
    ax8.grid(True, alpha=0.3)
    
    # 9. 年度汇总
    ax9 = fig.add_subplot(3, 3, 9)
    # 按年汇总
    year_data = weekly_data.iloc[offset:].copy()
    year_data['predicted'] = weekly_new
    yearly = year_data.groupby('year').agg({'cases': 'sum', 'predicted': 'sum'}).reset_index()
    x = range(len(yearly))
    width = 0.35
    ax9.bar([i - width/2 for i in x], yearly['cases'], width, label='Observed', color='steelblue')
    ax9.bar([i + width/2 for i in x], yearly['predicted'], width, label='Predicted', color='coral')
    ax9.set_xticks(x)
    ax9.set_xticklabels(yearly['year'].astype(int))
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Annual Cases')
    ax9.set_title('Annual Cases Comparison')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/figures/sei_seir_weekly.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  已保存: results/figures/sei_seir_weekly.png")
    
    # 保存结果
    results_df = pd.DataFrame({
        'week': weeks,
        'observed_cases': observed_cases,
        'predicted_cases': weekly_new,
        'Lambda_v': Lambda_v,
        'R0': R0_t,
        'N_v': N_v,
        'I_v': I_v,
        'I_h': I_h
    })
    results_df.to_csv('/root/wenmei/results/data/sei_seir_weekly_results.csv', index=False)
    print("  已保存: results/data/sei_seir_weekly_results.csv")
    
    # 保存Λ_v用于符号回归
    lambda_v_df = pd.DataFrame({
        'week': weeks,
        'temperature': weekly_data['temperature'].values[offset:],
        'humidity': weekly_data['humidity'].values[offset:],
        'precipitation': weekly_data['precipitation'].values[offset:],
        'bi': weekly_data['bi'].values[offset:],
        'Lambda_v': Lambda_v
    })
    lambda_v_df.to_csv('/root/wenmei/results/data/lambda_v_for_symbolic.csv', index=False)
    print("  已保存: results/data/lambda_v_for_symbolic.csv (用于符号回归)")
    
    # 打印模型总结
    print("\n" + "=" * 70)
    print("SEI-SEIR 模型总结")
    print("=" * 70)
    print(f"""
【模型结构】
蚊虫 (SEI):
  dS_v/dt = Λ_v(t) - b·β_v·S_v·I_h/N_h - μ_v·S_v
  dE_v/dt = b·β_v·S_v·I_h/N_h - σ_v·E_v - μ_v·E_v
  dI_v/dt = σ_v·E_v - μ_v·I_v

人群 (SEIR):
  dS_h/dt = -b·β_h·S_h·I_v/N_h
  dE_h/dt = b·β_h·S_h·I_v/N_h - σ_h·E_h
  dI_h/dt = σ_h·E_h - γ·I_h
  dR_h/dt = γ·I_h

【参数】
固定参数 (文献):
  μ_v = {params.mu_v} /天, β_v = {params.beta_v}, β_h = {params.beta_h}
  b = {params.b} /天, σ_v = {params.sigma_v} /天
  σ_h = {params.sigma_h} /天, γ = {params.gamma} /天

TCN预测参数:
  Λ_v(t) = μ_v × k × BI(t)
  k = {k_scale:.2e} (校准得到)

【性能】
  病例相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  趋势准确率: {trend_acc:.2%}
  BI预测 R²: {tcn_result['r2']:.4f}

【下一步: 符号回归】
  输入: lambda_v_for_symbolic.csv
  目标: 找到 Λ_v(T, H, P) 的解析表达式
""")
    
    return {
        'params': params,
        'k_scale': k_scale,
        'correlation': corr,
        'r2_log': r2_log,
        'R0': R0_t,
        'Lambda_v': Lambda_v
    }


if __name__ == '__main__':
    results = run_sei_seir_model()
