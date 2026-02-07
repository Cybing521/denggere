#!/usr/bin/env python3
"""
Phase 1 (v2): NN学习传播率β(T,H,R)
================================================
两步法 (参照PNAS实际做法):
  Step 1: 从病例数据反推月度传播率β(t)序列 (逆问题)
  Step 2: 训练NN拟合 β(t) = NN(T,H,R) (监督学习)
  → NN学到气象如何影响传播效率

再加:
  Step 3: 用NN预测的β重新跑SEIR验证
  Phase 2: 符号回归 → NN → 解析公式
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. 数据加载
# ============================================================

def load_data():
    log("\n[1] 加载数据...")
    case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
    df = case_df[(case_df['year'] >= 2006) & (case_df['year'] <= 2019)].copy().reset_index(drop=True)

    bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
    df['bi'] = df['bi'].interpolate(method='linear').fillna(df['bi'].mean())

    df['is_2014'] = (df['year'] == 2014)
    df['in_loss'] = ~df['is_2014']

    weather_raw = df[['temperature', 'humidity', 'precipitation']].values.astype(np.float32)
    w_min, w_max = weather_raw.min(0), weather_raw.max(0)
    w_range = np.maximum(w_max - w_min, 1e-8)
    weather_norm = (weather_raw - w_min) / w_range

    log(f"  数据: {df.year.min()}-{df.year.max()}, {len(df)}月")
    log(f"  总病例: {df.cases.sum():,} (2014={df[df.is_2014].cases.sum():,})")
    return df, weather_raw, weather_norm, w_min, w_max


# ============================================================
# 2. Step 1: 从病例数据反推β(t)
# ============================================================

def estimate_beta_series(df):
    """
    从病例数据反推月度传播势能β(t)

    月度SIR简化: cases(t) ≈ β(t) × M̂(t) × cases_pool(t-1)
    因此: β(t) ≈ cases(t) / (M̂(t) × cases_pool(t-1))

    β(t)反映: 单位蚊虫密度下的传播效率, 即环境条件对传播的影响
    """
    log("\n[Step 1] 从病例数据反推传播势能β(t)...")

    cases = df['cases'].values.astype(float)
    bi = df['bi'].values.astype(float)
    temp = df['temperature'].values

    # 蚊虫密度代理 (BI, 平滑)
    M_proxy = gaussian_filter1d(np.maximum(bi, 0.1), sigma=1)
    M_norm = M_proxy / (M_proxy.mean() + 1e-6)  # 归一化

    # 传染池: 上月+上上月病例 (考虑潜伏期和感染期滞后)
    cases_pool = np.ones_like(cases)
    for t in range(1, len(cases)):
        cases_pool[t] = max(cases[t-1] + 0.3 * (cases[t-2] if t >= 2 else 0), 1.0)

    # β(t) = cases(t) / (M_norm(t) × cases_pool(t))
    beta_raw = cases / (M_norm * cases_pool + 1e-6)

    # 平滑 + 裁剪极端值
    p95 = np.percentile(beta_raw[beta_raw > 0], 95) if (beta_raw > 0).any() else 1.0
    beta_clip = np.clip(beta_raw, 0, p95)
    beta_smooth = gaussian_filter1d(beta_clip, sigma=1.5)

    # 归一化到 [0, 1]
    beta_max = beta_smooth.max() + 1e-10
    beta_norm = beta_smooth / beta_max

    log(f"  β范围: [0, {beta_max:.4f}] → 归一化[0, 1]")
    log(f"  β>0.01月: {(beta_norm > 0.01).sum()}/{len(beta_norm)}")

    valid = ~df['is_2014'].values
    if np.std(beta_norm[valid]) > 0:
        corr_t, _ = pearsonr(temp[valid], beta_norm[valid])
        log(f"  β-温度相关(非2014): r={corr_t:.4f}")

    return beta_norm, M_norm, beta_max


# ============================================================
# 3. Step 2: 训练NN拟合β(t) = NN(T,H,R)
# ============================================================

class TransmissionNN(nn.Module):
    """NN(T,H,R) → β'(传播效率)"""
    def __init__(self, n_input=3, n_hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()   # 输出 ∈ (0, 1)
        )
    def forward(self, x):
        return self.net(x)


def train_nn(weather_norm, beta_target, in_loss_mask, n_epochs=2000, lr=0.005):
    """
    Step 2: 监督学习 — NN拟合反推的β(t)
    """
    log("\n[Step 2] 训练NN: (T,H,R) → β(t)")

    nn_model = TransmissionNN(n_input=3, n_hidden=16)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X = torch.from_numpy(weather_norm.astype(np.float32))
    y = torch.from_numpy(beta_target.astype(np.float32))
    mask = torch.from_numpy(in_loss_mask)

    best_loss, best_state = float('inf'), None
    losses = []

    for ep in range(n_epochs):
        nn_model.train()
        optimizer.zero_grad()

        pred = nn_model(X).squeeze()
        pred_m = pred[mask]
        y_m = y[mask]

        loss_mse = torch.mean((pred_m - y_m)**2)

        # 相关性损失
        if pred_m.std() > 1e-6:
            pn = (pred_m - pred_m.mean()) / (pred_m.std() + 1e-6)
            yn = (y_m - y_m.mean()) / (y_m.std() + 1e-6)
            loss_corr = -torch.mean(pn * yn)
        else:
            loss_corr = torch.tensor(0.0)

        loss = loss_mse + 0.5 * loss_corr
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

        if (ep + 1) % 400 == 0 or ep == 0:
            with torch.no_grad():
                p = nn_model(X).squeeze().numpy()
                t = beta_target
                m = in_loss_mask
                corr = np.corrcoef(t[m], p[m])[0,1] if np.std(p[m]) > 0 else 0
                r2 = r2_score(t[m], p[m]) if np.std(p[m]) > 0 else -999
            log(f"  Ep {ep+1:5d}: L={loss.item():.6f} r={corr:.4f} R²={r2:.4f}")

    nn_model.load_state_dict(best_state)
    log(f"  完成! best_loss={best_loss:.6f}")
    return nn_model, losses


# ============================================================
# 4. Step 3: 用NN预测的β重新跑SEIR验证
# ============================================================

def validate_with_seir(nn_model, weather_norm, df, M_norm, beta_max):
    """
    用NN的β预测病例 (简洁公式, 量纲一致)
    pred_cases(t) = β_nn(t) × M̂(t) × cases_pool(t-1)
    """
    log("\n[Step 3] 验证...")

    nn_model.eval()
    X = torch.from_numpy(weather_norm.astype(np.float32))
    with torch.no_grad():
        beta_nn_norm = nn_model(X).squeeze().numpy()

    beta_nn = beta_nn_norm * beta_max  # 反归一化
    cases = df['cases'].values.astype(float)

    # 用与Step 1一致的公式: cases(t) ≈ β(t) × M̂(t) × cases_pool(t-1)
    pred_cases = np.zeros(len(df))
    for t in range(1, len(df)):
        cases_pool = max(cases[t-1] + 0.3 * (cases[t-2] if t >= 2 else 0), 1.0)
        pred_cases[t] = beta_nn[t] * M_norm[t] * cases_pool

    # 评估
    lm = ~df['is_2014'].values
    lm[0] = False  # 第一个月无法预测
    if np.std(pred_cases[lm]) > 0:
        corr, p = pearsonr(cases[lm], pred_cases[lm])
        r2l = r2_score(np.log1p(cases[lm]), np.log1p(np.maximum(pred_cases[lm], 0)))
    else:
        corr, p, r2l = 0, 1, -999

    corr_all = np.corrcoef(cases[1:], pred_cases[1:])[0,1] if np.std(pred_cases[1:]) > 0 else 0

    log(f"  验证 (排除2014): r={corr:.4f}, R²(log)={r2l:.4f}")
    log(f"  验证 (全量):     r={corr_all:.4f}")

    return pred_cases, beta_nn, corr, r2l, corr_all


# ============================================================
# 5. 可视化
# ============================================================

def visualize(df, weather_raw, weather_norm, w_min, w_max,
              beta_target, beta_nn_raw, pred_cases, nn_model,
              M_proxy, nn_losses, corr_ex, r2l_ex, corr_all):
    log("\n[可视化]")
    oc = df['cases'].values.astype(float)
    pc = pred_cases
    lm = ~df['is_2014'].values
    c14 = df['is_2014'].values
    months = np.arange(len(oc))
    years = df['year'].values
    w_range = np.maximum(w_max - w_min, 1e-8)

    # NN全量预测
    nn_model.eval()
    X = torch.from_numpy(weather_norm.astype(np.float32))
    with torch.no_grad():
        beta_nn_norm = nn_model(X).squeeze().numpy()

    fig = plt.figure(figsize=(22, 24))

    # 1 - 病例拟合
    ax = fig.add_subplot(4,3,1)
    ax.plot(months, oc, 'b-', lw=2, label='Observed', marker='o', ms=2, alpha=.8)
    ax.plot(months, pc, 'r-', lw=2, label='SEIR+NN', marker='s', ms=2, alpha=.8)
    i14 = np.where(c14)[0]
    if len(i14): ax.axvspan(i14[0], i14[-1], alpha=.12, color='gray', label='2014')
    ax.set_title(f'Cases (excl 2014: r={corr_ex:.3f})'); ax.legend(fontsize=7)
    ax.set_ylabel('Cases'); ax.grid(True, alpha=.3)

    # 2 - log
    ax = fig.add_subplot(4,3,2)
    ax.semilogy(months, oc+1, 'b-', lw=2, marker='o', ms=2)
    ax.semilogy(months, pc+1, 'r-', lw=2, marker='s', ms=2)
    if len(i14): ax.axvspan(i14[0], i14[-1], alpha=.12, color='gray')
    ax.set_title(f'Log (R²={r2l_ex:.3f})'); ax.grid(True, alpha=.3)

    # 3 - 散点
    ax = fig.add_subplot(4,3,3)
    ax.scatter(oc[~c14], pc[~c14], alpha=.6, s=50, c='steelblue', edgecolors='w')
    ax.scatter(oc[c14], pc[c14], alpha=.8, s=80, c='red', marker='D', label='2014')
    mx = max(oc.max(), pc.max())*1.1
    ax.plot([0,mx],[0,mx],'k--',lw=2); ax.set_title('Scatter'); ax.legend(fontsize=7); ax.grid(True,alpha=.3)

    # 4 - β(t): 反推 vs NN
    ax = fig.add_subplot(4,3,4)
    ax.plot(months, beta_target, 'b-', lw=2, label='β estimated', alpha=.7)
    ax.plot(months, beta_nn_norm, 'r-', lw=2, label='β NN predicted', alpha=.8)
    ax.set_title('β(t): Estimated vs NN'); ax.legend(fontsize=8)
    ax.set_ylabel('β (normalized)'); ax.grid(True, alpha=.3)

    # 5 - β vs 温度
    ax = fig.add_subplot(4,3,5)
    sc = ax.scatter(weather_raw[:,0], beta_nn_norm, c=weather_raw[:,2], cmap='Blues', s=60, edgecolors='k', lw=.5)
    plt.colorbar(sc, ax=ax, label='Precip'); ax.set_xlabel('T (°C)'); ax.set_ylabel("β'")
    ax.set_title("NN: β' vs Temperature"); ax.grid(True, alpha=.3)

    # 6 - β vs 降水
    ax = fig.add_subplot(4,3,6)
    sc = ax.scatter(weather_raw[:,2], beta_nn_norm, c=weather_raw[:,0], cmap='Reds', s=60, edgecolors='k', lw=.5)
    plt.colorbar(sc, ax=ax, label='T °C'); ax.set_xlabel('Precip'); ax.set_ylabel("β'")
    ax.set_title("NN: β' vs Precipitation"); ax.grid(True, alpha=.3)

    # 7 - NN热力图
    ax = fig.add_subplot(4,3,7)
    Tg = np.linspace(0,1,50); Rg = np.linspace(0,1,50)
    TT, RR = np.meshgrid(Tg, Rg)
    gi = np.column_stack([TT.ravel(), np.full(TT.size,.5), RR.ravel()]).astype(np.float32)
    with torch.no_grad():
        gb = nn_model(torch.from_numpy(gi)).numpy().reshape(50,50)
    Tr = w_min[0]+TT*w_range[0]; Rr = w_min[2]+RR*w_range[2]
    im = ax.contourf(Tr, Rr, gb, levels=20, cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label="β'"); ax.set_xlabel('T °C'); ax.set_ylabel('Precip')
    ax.set_title("NN: β'(T, R) heatmap")

    # 8 - 年度
    ax = fig.add_subplot(4,3,8)
    dfe = pd.DataFrame({'year':years,'actual':oc,'predicted':pc})
    yr = dfe.groupby('year').agg({'actual':'sum','predicted':'sum'}).reset_index()
    x = range(len(yr)); w=.35
    ax.bar([i-w/2 for i in x], yr['actual'], w, label='Actual', color='steelblue')
    ax.bar([i+w/2 for i in x], yr['predicted'], w, label='Model', color='coral')
    ax.set_xticks(list(x)); ax.set_xticklabels(yr['year'], rotation=45, fontsize=7)
    ax.set_title('Annual'); ax.legend(fontsize=7); ax.grid(True,alpha=.3)

    # 9 - 蚊虫密度
    ax = fig.add_subplot(4,3,9)
    ax.plot(months, M_proxy, 'g-', lw=2); ax.set_title('Mosquito Density (BI proxy)')
    ax.set_ylabel('M̂(t)'); ax.grid(True, alpha=.3)

    # 10 - NN训练loss
    ax = fig.add_subplot(4,3,10)
    ax.semilogy(nn_losses, 'b-', lw=.3, alpha=.4)
    if len(nn_losses)>20: ax.semilogy(gaussian_filter1d(nn_losses,10), 'r-', lw=2)
    ax.set_title('NN Training Loss'); ax.grid(True,alpha=.3)

    # 11 - β(t)时间序列 + 温度
    ax = fig.add_subplot(4,3,11)
    ax.plot(months, beta_nn_norm, 'purple', lw=2.5)
    ax2 = ax.twinx()
    ax2.plot(months, weather_raw[:,0], 'orange', lw=1, alpha=.5)
    ax.set_ylabel("β'", color='purple'); ax2.set_ylabel('T °C', color='orange')
    ax.set_title("β'(t) and Temperature"); ax.grid(True, alpha=.3)

    # 12 - 总结
    ax = fig.add_subplot(4,3,12); ax.axis('off')
    ax.text(.02,.98, f"""
    v2: Two-Step NN-SEIR Model
    ===========================
    Step 1: Estimate β(t) from cases (inverse)
    Step 2: Train NN: (T,H,R) → β(t)
    Step 3: SEIR validation with NN-predicted β

    Performance (excl 2014):
      r = {corr_ex:.4f}, R²(log) = {r2l_ex:.4f}
    Performance (all):
      r = {corr_all:.4f}

    NN β' range: [{beta_nn_norm.min():.4f}, {beta_nn_norm.max():.4f}]

    Next: Phase 2 Symbolic Regression
    """, transform=ax.transAxes, fontsize=9, va='top', family='monospace',
       bbox=dict(boxstyle='round', fc='lightyellow', alpha=.8))

    plt.suptitle("Phase 1 (v2): Learning Transmission Rate β(T,H,R)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/figures/phase1_v2_transmission.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  图 → results/figures/phase1_v2_transmission.png")


def save_results(nn_model, df, weather_raw, weather_norm, w_min, w_max,
                 beta_target, pred_cases, beta_nn_raw):
    log("\n[保存]")
    w_range = np.maximum(w_max - w_min, 1e-8)
    nn_model.eval()

    # NN观测点
    obs_rec = []
    for j in range(len(weather_raw)):
        T,H,R = weather_raw[j]
        inp = torch.tensor([[(T-w_min[0])/w_range[0],(H-w_min[1])/w_range[1],
                              (R-w_min[2])/w_range[2]]], dtype=torch.float32)
        with torch.no_grad(): b = nn_model(inp).item()
        obs_rec.append({'temperature':T,'humidity':H,'precipitation':R,'oviposition_rate':b})
    pd.DataFrame(obs_rec).to_csv('/root/wenmei/results/data/nn_obs_output.csv', index=False)

    # NN网格
    recs = []
    for T in np.linspace(w_min[0],w_max[0],40):
        for H in np.linspace(w_min[1],w_max[1],10):
            for R in np.linspace(w_min[2],w_max[2],40):
                inp = torch.tensor([[(T-w_min[0])/w_range[0],(H-w_min[1])/w_range[1],
                                      (R-w_min[2])/w_range[2]]], dtype=torch.float32)
                with torch.no_grad(): b = nn_model(inp).item()
                recs.append({'temperature':T,'humidity':H,'precipitation':R,'oviposition_rate':b})
    pd.DataFrame(recs).to_csv('/root/wenmei/results/data/nn_grid_output.csv', index=False)

    torch.save(nn_model.state_dict(), '/root/wenmei/results/data/phase1_model.pt')
    np.savez('/root/wenmei/results/data/phase1_norm_params.npz', w_min=w_min, w_max=w_max)

    pd.DataFrame({
        'year':df.year.values,'month':df.month.values,
        'temperature':weather_raw[:,0],'humidity':weather_raw[:,1],'precipitation':weather_raw[:,2],
        'bi':df.bi.values,'cases_obs':df.cases.values,'cases_pred':pred_cases,
        'beta_estimated':beta_target,'beta_nn':beta_nn_raw,
        'is_2014':df.is_2014.values
    }).to_csv('/root/wenmei/results/data/phase1_predictions.csv', index=False)
    log("  已保存")


# ============================================================
# MAIN
# ============================================================

def main():
    log("="*70)
    log("Phase 1 (v2): 两步法学习传播率β(T,H,R)")
    log("="*70)
    log("""
    Step 1: 从病例反推β(t)序列 (逆问题)
    Step 2: NN拟合 β(t) = NN(T,H,R) (监督学习)
    Step 3: SEIR验证 — 用NN的β预测病例
    """)

    df, wr, wn, wmin, wmax = load_data()

    # Step 1: 反推β(t)
    beta_target, M_proxy, beta_max = estimate_beta_series(df)

    # Step 2: NN拟合β
    nn_model, nn_losses = train_nn(wn, beta_target, df['in_loss'].values,
                                    n_epochs=2000, lr=0.005)

    # Step 3: SEIR验证
    pred_cases, beta_nn_raw, corr_ex, r2l_ex, corr_all = validate_with_seir(
        nn_model, wn, df, M_proxy, beta_max)

    # 可视化
    nn_model.eval()
    X = torch.from_numpy(wn.astype(np.float32))
    with torch.no_grad():
        beta_nn_norm = nn_model(X).squeeze().numpy()

    visualize(df, wr, wn, wmin, wmax, beta_target, beta_nn_raw,
              pred_cases, nn_model, M_proxy, nn_losses, corr_ex, r2l_ex, corr_all)

    # 保存
    save_results(nn_model, df, wr, wn, wmin, wmax, beta_target, pred_cases, beta_nn_raw)

    # 分年度
    log("\n  年度对比:")
    for y in sorted(df.year.unique()):
        ym = df.year.values == y
        o, p = df.cases.values[ym], pred_cases[ym]
        yr = np.corrcoef(o, p)[0,1] if np.std(p) > 0 else 0
        tag = " ← 2014" if y == 2014 else ""
        log(f"  {y}: 实际={o.sum():>6.0f}  预测={p.sum():>8.0f}  r={yr:.4f}{tag}")

    log(f"\n{'='*70}")
    log(f"Phase 1完成! 排除2014: r={corr_ex:.4f}, R²(log)={r2l_ex:.4f}")
    log(f"{'='*70}")
    return nn_model

if __name__ == "__main__":
    main()
