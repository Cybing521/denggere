# 登革热传播动力学模型研究

## 基于神经网络耦合动力学与符号回归的传播机制发现

---

## 核心思路

本项目构建一个**三位一体**的登革热传播建模框架：

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ① 动力学模型 (SEI-SEIR) ← 论文主体/骨架               │
│     蚊虫: dS_v/dt = Λ_v(T) - μ_v·S_v - ...            │
│     人群: dI_h/dt = β(t)·M̂/N·S_h - γ·I_h             │
│                      ↑                                  │
│  ② 机器学习 (神经网络) ← 替代未知的传播率β              │
│     β(t) = NN(T, H, R)                                 │
│     训练: ODE数值解 vs 实际病例 → 反向传播               │
│                      ↓                                  │
│  ③ 符号回归 ← 把NN黑箱翻译成解析公式                    │
│     NN(T,H,R) → β = f(T,H,R) = 显式数学表达式          │
│                                                         │
│  最终产物: 完全可解释的动力学模型                         │
│     dI_h/dt = f(T,H,R)·M̂/N·S_h - γ·I_h               │
└─────────────────────────────────────────────────────────┘
```

### 三者缺一不可

| 角色 | 做什么 | 为什么需要 |
|------|--------|-----------|
| **动力学模型** | 提供生物学框架 (SEI-SEIR) | 保证结果有物理/生物学意义，论文主体 |
| **机器学习** | 发现未知的传播率函数 β(T,H,R) | β与气象的关系形式未知，不能人为预设 |
| **符号回归** | 把NN黑箱翻译成数学公式 | NN不可解释，论文需要显式公式 |

### 与参考文献的关系

| 参考文献 | 贡献 | 我们如何结合 |
|---------|------|------------|
| **PNAS (Li et al. 2019)** | SIR + 蚊虫密度驱动 + 样条β'(t) | 借用框架，用NN替代样条 |
| **PLoS Comp Bio (Zhang et al. 2024)** | NN嵌入ODE + 符号回归 | 借用NN+ODE耦合训练方法 |
| **本研究** | 两者结合 | PNAS框架 + Zhang方法 = NN学习传播率 + 符号回归 |

**创新点**: PNAS的β'(t)是样条曲线（只随时间变化，不知道为什么变化）。我们用NN替代，输入是气象变量(T,H,R)，所以能回答"温度27°C、降水5mm时传播效率是多少"。再通过符号回归获得显式公式，比PNAS更有机理性和预测力。

---

## 模型方程

### 蚊虫动力学 (SEI)

蚊虫密度 M̂(t) 可从BI/MOI监测数据获得，或用GAM从气象预测（参照PNAS）。

### 人群动力学 (SEIR)

$$\frac{dS_h}{dt} = -\frac{\text{NN}(T,H,R) \cdot \hat{M}(t)}{N_h} \cdot S_h$$

$$\frac{dE_h}{dt} = \frac{\text{NN}(T,H,R) \cdot \hat{M}(t)}{N_h} \cdot S_h - \sigma_h E_h$$

$$\frac{dI_h}{dt} = \sigma_h E_h - \gamma I_h$$

$$\frac{dR_h}{dt} = \gamma I_h$$

其中 **NN(T, H, R)** 是神经网络学习的**传播效率**（per-mosquito vector efficiency），替代PNAS中的样条函数β'(t)。

### 两阶段流程

- **Phase 1**: NN嵌入ODE → 拟合2006-2019病例数据 → NN间接学会 β(T,H,R)
  - 2014年：ODE连续跑，但2014年不参与loss（极端暴发，非气象因素主导）
- **Phase 2**: 符号回归 → 发现 β(T,H,R) 的解析公式 → 完全可解释的动力学模型

---

## 数据来源

| 数据 | 来源 | 引用 |
|------|------|------|
| 登革热月度病例 + 气象 | **CCM14 数据集** | GitHub: [xyyu001/CCM14](https://github.com/xyyu001/CCM14) |
| 蚊虫BI/MOI (月度) | **CCM14 数据集** | 同上 |
| 蚊虫MOI (半月度, 2016-2019) | **CCM14 数据集** | 同上 |
| 补充气象 (日度) | **Open-Meteo API** | https://open-meteo.com/ |

数据区域: 广东省（省级病例）/ 广州市（蚊虫监测+气象）  
数据时段: 2005-2019年

---

## 版本记录

### v1: NN耦合蚊虫种群模型 (2025-02-06)

- 参照Zhang et al. 2024，NN替代产卵率
- 问题：产卵率是蚊虫生态的未知项，不是疾病传播的未知项
- 结果：BI拟合 r=0.64，病例 r=0.59（排除2014后）
- 文件夹: `v1_nn_coupled_dynamics/`

### v2: NN耦合传播率模型 (进行中)

- 结合PNAS框架 + Zhang方法
- NN替代传播率β(T,H,R)（疾病传播的核心未知项）
- 蚊虫密度从BI/MOI数据获得
- 2014年: ODE连续跑，不参与loss
- 文件夹: `v2_nn_transmission_rate/` (待创建)

---

## 项目结构

```
wenmei/
├── data/
│   ├── guangdong_dengue_cases.csv      # 病例+气象 (CCM14)
│   ├── BI.csv                          # 蚊虫监测 (CCM14)
│   ├── public_sources/                 # 公开数据源
│   │   ├── ccm14_guangdong_MOI_2016_2019.xlsx
│   │   ├── ccm14_guangdong_BI_2016_2019.xlsx
│   │   ├── openmeteo_guangzhou_daily_2005_2019.csv
│   │   └── openmeteo_guangzhou_monthly_2005_2019.csv
│   └── data_Guangdong.xlsx             # 原始数据 (= CCM14)
├── src/
│   ├── phase1_coupled_model.py         # Phase 1: NN耦合动力学
│   └── phase2_formula_discovery.py     # Phase 2: 符号回归
├── paper/                              # 参考文献
├── v1_nn_coupled_dynamics/             # v1版本完整结果
│   ├── code/
│   ├── results/
│   └── report/
├── v2_nn_transmission_rate/            # v2版本 (进行中)
└── README.md                           # 本文件
```

## 参考文献

1. **Li R, et al. (2019)**. Climate-driven variation in mosquito density predicts the spatiotemporal dynamics of dengue. *PNAS*, 116(9): 3624-3629. — 动力学框架

2. **Zhang M, Wang X, Tang S (2024)**. Integrating dynamic models and neural networks to discover the mechanism of meteorological factors on Aedes population. *PLoS Computational Biology*, 20(9): e1012499. — NN+ODE+符号回归方法

3. **CCM14 Dataset**. https://github.com/xyyu001/CCM14 — 数据来源
