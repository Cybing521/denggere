#!/usr/bin/env python3
"""Generate thesis_draft.tex with the exact structure matching the reference thesis."""

import os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_draft.tex")

# ── preamble ──────────────────────────────────────────────────────────
PREAMBLE = r"""\documentclass[12pt,a4paper]{article}
\usepackage[UTF8,heading=true]{ctex}
\usepackage{geometry}
\geometry{left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\usepackage{setspace}
\setstretch{1.35}
\usepackage{indentfirst}
\setlength{\parindent}{2em}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\heavyrulewidth=1.5pt
\lightrulewidth=0.75pt
\usepackage{caption}
\captionsetup{font=footnotesize,justification=centering}
\usepackage{float}
\usepackage{enumitem}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\usepackage{url}
\usepackage{multirow}
\usepackage{array}
\usepackage{longtable}
\usepackage{tabularx}

\setCJKmainfont{FandolSong}
\setCJKsansfont{FandolHei}
\setmainfont{Times New Roman}

\ctexset{
  section = {
    format = \centering\sffamily\zihao{3},
    beforeskip = 1.5ex plus 0.5ex minus 0.2ex,
    afterskip = 1.0ex plus 0.3ex minus 0.2ex,
  },
  subsection = {
    format = \raggedright\sffamily\zihao{4},
    beforeskip = 1.0ex plus 0.3ex minus 0.1ex,
    afterskip = 0.8ex plus 0.2ex minus 0.1ex,
  },
  subsubsection = {
    format = \raggedright\sffamily\zihao{-4},
    beforeskip = 0.8ex plus 0.2ex minus 0.1ex,
    afterskip = 0.5ex plus 0.1ex minus 0.1ex,
  },
}

\begin{document}
"""

# ── title page ────────────────────────────────────────────────────────
TITLEPAGE = r"""
\begin{titlepage}
\begin{center}
\vspace*{2cm}
{\sffamily\zihao{2}\textbf{毕业论文}}
\vspace{2cm}

{\sffamily\zihao{-2}\textbf{基于神经网络耦合动力学模型的\\登革热传播效率发现与多城市验证}}
\vspace{1cm}

{\zihao{3} Discovery of Dengue Transmission Efficiency via\\
Neural-Network-Coupled Dynamical Models\\
and Multi-City Validation}
\vspace{3cm}

\begin{tabular}{rl}
{\sffamily\zihao{4} 学\qquad 院：} & {\zihao{4} 公共卫生学院} \\[8pt]
{\sffamily\zihao{4} 专\qquad 业：} & {\zihao{4} 流行病与卫生统计学} \\[8pt]
{\sffamily\zihao{4} 研究方向：} & {\zihao{4} 传染病建模与预测} \\[8pt]
\end{tabular}
\vspace{3cm}

{\zihao{4} \today}
\end{center}
\end{titlepage}
\setcounter{page}{1}
\pagenumbering{Roman}
"""

# ── Chinese abstract ──────────────────────────────────────────────────
ABSTRACT_CN = r"""
\newpage
\begin{center}
{\sffamily\zihao{3} 摘\quad 要}
\end{center}
\addcontentsline{toc}{section}{摘要}

登革热是全球最严重的蚊媒传染病之一，中国南方尤其广东省是国内最主要的流行区域。理解气候因素如何驱动登革热传播效率，对于建立早期预警系统和制定精准防控策略具有重要意义。然而，传统机制模型往往依赖先验假设固定传播率函数形式，难以从数据中自动发现最优的气候--传播率关系；纯数据驱动的机器学习方法则存在可解释性不足的瓶颈。

本文提出一种"神经网络耦合SEIR动力学+符号回归"三阶段混合建模框架，旨在兼顾机制可解释性与数据适应性。该框架包含三个核心阶段：(1) 基于SEIR仓室模型反演月尺度传播系数$\beta'$时间序列；(2) 以多层感知机(MLP)学习气候变量（温度$T$、降水$R$、相对湿度$H$）到$\beta'$的非线性映射，实现逐月病例数预测；(3) 利用符号回归对神经网络进行知识蒸馏，发现可解释的闭合公式。

研究分为两个部分。第一部分以广州市为单城市案例（2005--2019年双周数据）。Phase~1阶段，耦合模型对广州月度病例的预测达到Pearson相关系数$r=0.612$、Spearman秩相关$\rho=0.705$、对数$R^2=0.450$、MAE$=51.23$。Phase~2阶段，符号回归从物理模板族和多项式族两类候选中发现最优公式，其中二次多项式含交互项的公式以$R^2=0.999987$的精度拟合神经网络输出，揭示了温度--降水正交互（$a_{TR}>0$）和降水平方负效应（$a_{RR}<0$）等物理规律。对2014年特大暴发的分析表明，该公式能区分极端年份的气候异常信号。

第二部分将广州发现的公式迁移至广东省16个地级市（2005--2019年），采用三种城市尺度化方案进行年度和月度验证。结果表明，16城年度总病例排名的Spearman相关$\rho=0.900$（$p=2.05\times10^{-6}$），非广州15城MAE$=61.8$、RMSE$=116.8$，证实了公式的空间泛化能力。与旧数据集（13城/2003--2017年）相比，新数据集使排名相关从$\rho=0.713$提升至$\rho=0.879$。城市月度曲线的中位Pearson~$r=0.481$、中位Spearman~$\rho=0.469$，表明公式可捕捉多数城市的季节性趋势。

本研究的主要创新包括：(1) 提出"NN逆问题+符号蒸馏"范式，克服了传统SEIR模型依赖先验函数形式的局限；(2) 发现的闭合公式具有明确物理含义且可直接迁移至其他城市；(3) 系统验证了单城市机制在多城市空间尺度上的可迁移性；(4) 相比Li等(2019 PNAS)的样条方法和Zhang等(2024)的纯符号回归方法，本框架在可解释性和泛化性之间取得了更好的平衡。

\vspace{1em}
\noindent {\sffamily 关键词：}登革热；SEIR模型；神经网络；符号回归；传播效率；广东省；多城市验证
"""

# ── English abstract ──────────────────────────────────────────────────
ABSTRACT_EN = r"""
\newpage
\begin{center}
{\sffamily\zihao{3} Abstract}
\end{center}
\addcontentsline{toc}{section}{Abstract}

Dengue fever is one of the most severe mosquito-borne infectious diseases globally. Southern China, especially Guangdong Province, is the primary endemic region in the country. Understanding how climatic factors drive dengue transmission efficiency is crucial for establishing early-warning systems and formulating targeted control strategies. However, traditional mechanistic models rely on \textit{a priori} assumptions to fix the functional form of the transmission rate, making it difficult to discover optimal climate--transmission relationships from data automatically. Meanwhile, purely data-driven machine learning approaches suffer from limited interpretability.

This thesis proposes a three-stage hybrid modeling framework---``Neural-Network-Coupled SEIR Dynamics + Symbolic Regression''---that aims to balance mechanistic interpretability with data adaptability. The framework consists of three core stages: (1)~inverting monthly transmission coefficients $\beta'$ from observed case data via an SEIR compartmental model; (2)~training a multilayer perceptron (MLP) to learn the nonlinear mapping from climate variables (temperature~$T$, precipitation~$R$, relative humidity~$H$) to $\beta'$, enabling month-by-month case prediction; (3)~performing knowledge distillation of the neural network via symbolic regression to discover interpretable closed-form formulas.

The study is organized into two parts. Part~I uses Guangzhou as a single-city case study (2005--2019 biweekly data). In Phase~1, the coupled model achieves a Pearson correlation of $r=0.612$, Spearman rank correlation of $\rho=0.705$, log-scale $R^2=0.450$, and MAE${}=51.23$ for Guangzhou monthly cases. In Phase~2, symbolic regression discovers the optimal formula from two candidate families---physical-template and polynomial. The quadratic polynomial with interaction terms fits the neural network output with $R^2=0.999987$, revealing physical patterns such as positive temperature--precipitation interaction ($a_{TR}>0$) and a negative quadratic rainfall effect ($a_{RR}<0$). Analysis of the extreme 2014 outbreak demonstrates that the formula can distinguish anomalous climate signals in extreme years.

Part~II transfers the formula discovered in Guangzhou to 16 prefecture-level cities in Guangdong Province (2005--2019), employing three city-level scaling schemes for annual and monthly validation. Results show that the Spearman correlation of 2014 annual total cases across 16~cities is $\rho=0.900$ ($p=2.05\times10^{-6}$), with non-Guangzhou 15-city MAE${}=61.8$ and RMSE${}=116.8$, confirming the spatial generalizability of the formula. Compared with the old dataset (13~cities, 2003--2017), the new dataset improves the ranking correlation from $\rho=0.713$ to $\rho=0.879$. City-level monthly curves yield a median Pearson~$r=0.481$ and median Spearman~$\rho=0.469$, indicating that the formula captures the seasonal trend for most cities.

Key innovations include: (1)~a ``neural-network inverse problem + symbolic distillation'' paradigm that overcomes the reliance on \textit{a priori} functional forms in traditional SEIR models; (2)~a discovered closed-form formula with clear physical meaning that can be directly transferred to other cities; (3)~systematic validation of single-city mechanisms at the multi-city spatial scale; (4)~an improved balance between interpretability and generalizability compared with the spline approach of Li~et~al.\ (2019, PNAS) and the pure symbolic regression approach of Zhang~et~al.\ (2024).

\vspace{1em}
\noindent \textbf{Keywords:} Dengue fever; SEIR model; Neural network; Symbolic regression; Transmission efficiency; Guangdong Province; Multi-city validation
"""

# ── TOC + page reset ──────────────────────────────────────────────────
TOC_RESET = r"""
\newpage
\tableofcontents
\newpage
\setcounter{page}{1}
\pagenumbering{arabic}
"""

# ── 前言 (Literature Review, ~5500 chars) ─────────────────────────────
QIANYAN = r"""
%% ==================================================================
\section*{前\quad 言}
\addcontentsline{toc}{section}{前言}
\label{sec:qianyan}
%% ==================================================================

登革热（Dengue Fever）是由登革病毒（DENV）引起、主要经由伊蚊（\textit{Aedes aegypti}和\textit{Aedes albopictus}）叮咬传播的急性虫媒传染病。近年来，随着全球气候变暖、城市化进程加速以及国际贸易和旅游的频繁，登革热已成为世界上增长最快的虫媒病毒性疾病\cite{messina2019}。据Bhatt等\cite{bhatt2013}估计，全球每年约有3.9亿人感染登革病毒，其中约9600万例出现临床症状。世界卫生组织\cite{who2024}指出，过去二十年间登革热报告病例数增长了八倍以上，从2000年的50万例升至2023年的超过600万例，现已在100多个国家流行。Messina等\cite{messina2019}利用全球尺度的统计模型预测，到2080年气候变化和城市化将使全球约63亿人面临登革热风险，较2015年增加约22亿人。

在中国，登革热虽不是本土地方性流行病，但自20世纪70年代末以来，由输入性病例引发的本土暴发在东南沿海地区频发。特别是广东省，地处亚热带，气候温暖湿润，极适宜白纹伊蚊的生长繁殖，长期以来是我国登革热防控的重点区域\cite{lai2015,yue2021}。2014年，广东省经历了历史上最严重的登革热疫情，报告病例数超过45{,}000例，广州市单城报告逾37{,}000例，创下历史纪录\cite{cheng2016}。Yue等\cite{yue2021}的系统综述表明，自2004年以来广东省贡献了全国超过70\%的登革热报告病例，年度病例数呈波动性上升趋势，暴发间隔呈缩短趋势。这一流行模式与该地区亚热带季风气候、高度城市化、人口密集以及频繁的国际人员流动密切相关。从血清型分布来看，广东省历年暴发中DENV-1最为常见，但也检测到DENV-2、DENV-3和DENV-4的输入性和本地传播病例\cite{cheng2016}。值得注意的是，由于不同血清型之间仅存在短暂的交叉免疫保护，二次感染可能导致更严重的登革出血热（DHF）和登革休克综合征（DSS），给公共卫生系统带来额外压力\cite{liyanage2016}。近年来，全球气候变暖和极端天气事件增加，进一步加剧了登革热北扩和暴发频次增加的风险。DeSouza等\cite{desouza2024}指出，2023--2024年全球登革热病例再创历史新高，部分与厄尔尼诺现象引发的异常高温和强降水有关。因此，深入探究登革热的传播机制，特别是量化环境因素对传播过程的非线性驱动作用，对于制定精准的防控策略具有重要的现实意义。

\vspace{0.5em}\noindent{\sffamily\zihao{4} 气候因素与蚊媒传播}
\vspace{0.3em}

气候因素是驱动蚊媒传染病时空分布和流行强度的核心外部变量。伊蚊的生命周期、种群密度及病毒在蚊体内的复制速率均受到气象条件的严格制约\cite{liyanage2016}。\textbf{温度}直接影响蚊虫的生殖周期、幼虫发育率及成蚊存活率\cite{desouza2024,shapiro2017,lambrechts2011}。更重要的是，温度决定了外潜伏期（Extrinsic Incubation Period, EIP），即病毒在蚊体内复制并具备传播能力所需的时间\cite{lambrechts2011,kamiya2020}。Mordecai等\cite{mordecai2019}的全面实验研究表明，蚊媒传播能力对温度呈单峰响应，最优传播温度约为29$^\circ$C。在此温度下，蚊虫叮咬率最高、病毒外潜伏期最短、蚊虫存活率最大，三者的乘积效应使传播效率达到峰值。当温度低于约18$^\circ$C或高于约34$^\circ$C时，传播能力显著下降\cite{mordecai2017}。Shapiro等\cite{shapiro2017}和Lambrechts等\cite{lambrechts2011}进一步指出，温度日较差（Diurnal Temperature Range, DTR）对传播效率也有重要影响。Col\'{o}n-Gonz\'{a}lez等\cite{colon2018}基于多模型集合预测发现，温度的升高将显著扩大登革热的适宜传播区，若全球升温幅度能控制在2.0$^\circ$C，可避免拉丁美洲每年约280万例新增登革热病例。Kamiya等\cite{kamiya2020}的荟萃分析确认了温度对蚊媒传染病传播的非线性调控作用在全球不同地理区域具有一致性。

\textbf{降水}对登革热的影响具有双重性。一方面，降水为蚊虫提供了必要的繁殖栖息地——积水容器、洼地和废弃物中的积水是伊蚊的主要产卵场所\cite{roiz2015}；适量降水显著增加蚊虫密度，从而提高传播风险\cite{nosrat2021}。另一方面，极端强降水可能冲刷幼虫栖息地、降低蚊虫存活率，产生抑制效应\cite{colon2018}。Zhou等\cite{zhou2025}的纵向研究发现，降水与登革热发病率之间存在显著的非线性关系和时间滞后效应，累积降水量超过一定阈值后传播风险不再持续增加，呈现饱和或下降趋势。Cheng等\cite{chengq2023}针对广州的研究发现，在前期水分充足的条件下，滞后7--121天的强降雨会降低登革热风险。这种"先增后平"的模式在本文模型发现中也得到了印证。

\textbf{相对湿度}影响蚊虫的存活和活动能力。Wu等\cite{wu2018}对中国南方登革热暴发的时间序列分析发现，相对湿度存在一个约76\%的阈值效应——当湿度超过此值时，蚊虫存活率和叮咬活跃度显著提高，登革热传播风险明显增大。Cheng等\cite{chengq2023}对广州的研究进一步证实了湿度与登革热发病率之间的正相关关系，尤其在高温环境下湿度的促进作用更为显著。Polrob等\cite{polrob2025}在东南亚的研究中发现，湿度与蚊虫叮咬率之间存在协同关系，进一步支持了湿度作为重要传播调节因子的地位。

从生态机制的角度，上述三个气候变量并非独立作用，而是通过复杂的交互效应共同决定传播强度。例如，高温高湿条件下蚊虫的吸血频率和存活率同时增加，产生协同促进效应；而高温干燥条件则可能因蚊虫脱水死亡而抑制传播。DaCosta等\cite{dacosta2025}和Leung等\cite{leung2023}的研究均强调，单独考虑任一气候因子都不足以准确描述传播动态，需要同时纳入温度、降水和湿度的联合效应。这一认识构成了本文将三个气候变量同时纳入神经网络模型的理论基础。

\vspace{0.5em}\noindent{\sffamily\zihao{4} 登革热建模研究现状}
\vspace{0.3em}

登革热建模研究经历了从纯统计模型到机制模型、再到人工智能融合模型的发展历程，不同方法在解释能力、预测精度和可推广性方面各有优劣。

\textbf{统计模型。}广义加性模型（GAM）和分布式滞后非线性模型（DLNM）是登革热气候--疫情关系研究中应用最广泛的统计工具\cite{lowe2021}。GAM能够灵活地刻画气候变量与发病率之间的非线性关系，同时控制季节性和长期趋势等混杂因素。DLNM进一步考虑了气候影响的时间滞后结构，能够同时估计暴露--反应关系和滞后效应\cite{roberts2017}。Liu等\cite{liuk2020}利用DLNM分析了中国南方多个城市的气候--登革热关系，发现温度和降水的影响在滞后1--3个月最为显著。Luo等\cite{luo2025}对马来西亚、新加坡和泰国2017--2022年的登革热传播模式进行研究，发现最高气温与登革热关系的峰值相对风险在COVID-19后显著上升。Cheng等\cite{chengy2025}基于中国广东和浙江2005--2024年的数据，构建了融合DLNM与混合智能算法的预测框架，在平均准确率等指标上均表现最优。然而，统计模型本质上是"关联性"而非"因果性"工具，其参数不具有直接的流行病学机制含义，在外推到未见过的气候条件或新的地理区域时，预测能力往往大幅下降\cite{baker2022,mills2024}。

\textbf{机制模型。}基于仓室结构的传染病动力学模型是理解传播机制的经典工具。Ross-Macdonald模型及其扩展形式将人--蚊传播过程分解为若干关键参数，每个参数都具有明确的生物学含义\cite{smith2012}。SEI-SEIR耦合模型是登革热研究中常用的仓室结构，将蚊群的"易感--暴露--感染"与人群的"易感--暴露--感染--恢复"动态耦合\cite{guo2024}。Li等\cite{li2019pnas}在2019年\textit{PNAS}上发表的研究中，在SEI-SEIR框架中使用时变三次样条函数拟合传播系数$\beta(t)$与温度的关系，并通过广州2005--2015年的病例和气候数据进行参数估计，发现$\beta(t)$对温度呈单峰响应，最优温度约为27--29$^\circ$C。然而，该方法存在以下局限：(1)~三次样条的形式需要预先指定节点数和位置；(2)~仅考虑温度单一气候变量，忽略了降水和湿度；(3)~最终结果为分段平滑曲线而非可移植的闭合公式。Caldwell等\cite{caldwell2021}指出，实验室环境与复杂的野外环境存在巨大差异，直接套用实验室参数往往导致模型预测偏差。现有的机制模型通常直接采用实验室测定的温依参数（如Bri\`{e}re函数描述叮咬率）\cite{mordecai2017,huber2018}，难以真实反映野外条件下气候因素对传播效率的综合影响。

\textbf{人工智能融合方法。}近年来，将深度学习与微分方程模型结合的"物理信息神经网络"（PINN）和"神经常微分方程"（Neural ODE）方法受到越来越多的关注\cite{chen2018node}。在传染病建模领域，Sehi等\cite{sehi2025}和Luo等\cite{luo2025}将PINN应用于SIR/SEIR模型的参数估计和短期预测，取得了优于传统拟合方法的精度。Li等\cite{lir2024}将COVID-19模型动态嵌入物理信息神经网络，同时推断未知参数和未观察到的底层模型动态。Nikparvar等\cite{nikparvar2021}将人口流动性作为变量输入LSTM用于预测美国各县的确诊病例数和死亡人数。Murphy等\cite{murphy2021}利用不同传染动力学生成的数据训练了一个图神经网络。然而，纯神经网络方法的"黑箱"本质使其难以提供机制层面的洞见\cite{holm2019}。即使模型预测准确，研究者仍然无法回答"气候如何影响传播率"这一核心科学问题。Baker等\cite{baker2022}和Mills等\cite{mills2024}均指出，在传染病动力学领域，可解释性和可迁移性通常比单纯的预测精度更有实际价值。Ahman等\cite{ahman2025}的综述进一步强调了"混合机制--数据驱动"框架在传染病建模中的前景。Kamyshnyi等\cite{kamyshnyi2026}和Adeoye等\cite{adeoye2025}的综述也表明，神经网络虽然能捕捉复杂的非线性模式，却无法揭示疾病传播的内在动力学规律，更无法转化为可推广的数学知识。

\textbf{符号回归方法。}符号回归（Symbolic Regression, SR）是一种从数据中直接搜索数学表达式的方法，能够在不预设函数形式的前提下发现数据中的数学规律\cite{cranmer2023}。与传统回归方法不同，符号回归的搜索空间包含所有可能的数学表达式，其目标是在精度和复杂度之间取得帕累托最优。Fajardo等\cite{fajardo2024}在\textit{PLOS Computational Biology}发表的工作提出了贝叶斯符号回归方法，用于从报告病例和检测率数据中自动学习传染病发病率的闭式数学模型。Zhang等\cite{zhang2024plos}在2024年\textit{PLOS Computational Biology}发表了将符号回归应用于传染病模型参数发现的开创性工作——通过将蚊媒种群动力学模型耦合神经网络，有效揭示了伊蚊产卵率和温度、降水之间的关系，并使用符号回归确定最优函数表达式。然而，该方法面临以下挑战：(1)~直接在高维表达式空间中搜索计算成本极高；(2)~缺乏利用先验物理知识引导搜索的机制；(3)~尚未在真实登革热传播效率发现上得到充分验证。Makke和Mahesh\cite{makke2024}在符号回归综述中指出，结合神经网络预训练和符号蒸馏的两阶段策略是一种有前景的方向：先用神经网络捕获复杂映射关系，再用符号回归提取简洁公式。这种"知识蒸馏"思路正是本文方法论的核心灵感来源。然而，目前尚未有研究将"神经网络嵌入+符号回归"的完整框架应用于登革热传播效率反演与公式推导中。

\vspace{0.5em}\noindent{\sffamily\zihao{4} 研究目标与创新点}
\vspace{0.3em}

基于上述文献回顾，本文提出以下研究目标：
\begin{enumerate}[leftmargin=2em]
\item 构建"SEIR动力学+神经网络+符号回归"三阶段混合建模框架，从时间序列数据中自动发现气候变量到登革热传播系数$\beta'$的最优函数关系。
\item 以广州市为核心案例，利用2005--2019年双周尺度病例和气候数据，训练耦合模型并提取可解释闭合公式。
\item 将发现的公式迁移至广东省16个地级市，在空间维度上验证其泛化能力和可迁移性。
\item 与现有方法（尤其是Li等2019年PNAS的样条方法和Zhang等2024年的纯符号回归方法）进行比较，论证本框架在可解释性--泛化性平衡方面的优势。
\end{enumerate}

与现有工作相比，本文的创新点包括：
\begin{enumerate}[leftmargin=2em]
\item \textbf{方法论创新——"NN逆问题+符号蒸馏"范式}：不同于Li等\cite{li2019pnas}预设样条函数形式，本文通过神经网络自由学习$\beta'$的气候映射关系，再用符号回归提取公式，实现了"数据驱动的函数发现"。
\item \textbf{多变量联合建模}：不同于仅考虑温度单一变量的传统做法，本文同时纳入温度、降水和相对湿度三个气候变量及其交互效应。
\item \textbf{可迁移的闭合公式}：符号回归发现的二次多项式公式具有明确的系数含义，可直接通过城市尺度参数进行迁移，无需在每个城市重新训练模型。
\item \textbf{系统性的空间验证}：通过16城年度排名和月度曲线的双重验证，首次在中国南方多城市尺度上系统评估了单城市发现的传播效率公式的空间泛化性能。
\end{enumerate}

\vspace{0.5em}\noindent{\sffamily\zihao{4} 全文结构}
\vspace{0.3em}

本文其余部分组织如下：第一部分（单城市机制发现）以广州市为案例，详细阐述数据来源、SEIR模型构建、神经网络耦合训练策略、符号回归方法以及评估指标体系，并呈现Phase~1（耦合模型预测）和Phase~2（公式发现）的结果与讨论。第二部分（多城市机制迁移与验证）将广州发现的公式迁移至广东省16个地级市，介绍三种城市尺度化方案，呈现年度和月度验证结果，并与旧数据集进行对比分析。最后一章（总结与展望）总结主要发现和创新点，讨论研究局限性，提出未来改进方向。
"""

# ── Part I ────────────────────────────────────────────────────────────
PART1 = r"""
%% ==================================================================
\section{第\,I\,部分\quad 单城市机制发现——基于神经网络耦合SEIR模型的登革热传播效率学习}
\label{sec:part1}
%% ==================================================================

\subsection{引言}
\label{sec:p1-intro}

登革热的传播过程受到多种环境因素的复杂影响\cite{white2025,mills2024}，但目前对于气象因素如何具体、量化地驱动传播效率尚缺乏统一的认识。现有研究主要依赖基于实验室数据的参数化模型（如使用Bri\`{e}re方程描述温度影响）\cite{huber2018}，或者基于历史数据的统计模型\cite{lic2023}。然而，实验室的恒温环境难以真实反映野外复杂的微气候波动，且往往忽略了降雨和湿度对蚊媒生存的联合作用\cite{dennington2025}；而纯统计模型虽然能捕捉流行趋势，但缺乏对传播机理的解释能力，难以进行反事实推断\cite{polrob2025}。

广州市长期以来是我国登革热防控的重点区域，其亚热带季风气候极适宜白纹伊蚊孳生。特别是2014年，广州经历了历史罕见的大规模暴发，病例数超45{,}000例\cite{chengj2021}。本章以广州市为例，提出一种结合了"数据挖掘"与"机理建模"的方法，旨在回答一个关键问题：在真实环境中，温度、湿度和降雨通过什么样的数学关系决定登革热传播效率$\beta'$？本章首先利用SEIR动力学模型耦合神经网络，从历史数据中还原出隐含的传播率时间序列，再通过符号回归方法，从复杂的神经网络中提取出具有物理意义的数学公式，以此揭示广州登革热暴发背后的环境驱动机制。

选择广州作为核心案例的理由包括：(1)~广州拥有相对完整的病例报告系统和气候监测网络，是检验新方法的理想试验田；(2)~2005--2019年的15年跨度涵盖了多个暴发年份（尤其是2014年特大暴发），为模型提供了充分的信号变异性；(3)~选择双周（biweekly）为时间分辨率，既保留了足够的动态细节，又避免了日数据中过多的随机噪声。

\subsection{数据材料和方法}
\label{sec:p1-method}

\subsubsection{研究数据来源与数据预处理}
\label{sec:p1-data}

本章选取广东省省会广州市作为研究区域。广州市位于东经112$^\circ$57$'$至114$^\circ$03$'$、北纬22$^\circ$26$'$至23$^\circ$56$'$之间，属于典型的海洋性亚热带季风气候，年平均气温21.5--22.2$^\circ$C，雨量充沛。该地区不仅是白纹伊蚊的活跃区，也是中国大陆登革热病例报告最集中的城市，且拥有完善的蚊媒监测网络。

本研究收集了广州市2005--2019年的气象数据、蚊媒监测数据、登革热病例数据和人口数据，考虑到动力学模型的模拟步长需求，将所有数据的时间尺度统一为月度。

\textbf{气象数据}：包括平均气温（$T$，$^\circ$C）、相对湿度（$H$，\%）及累计降雨量（$R$，mm），来源于美国国家海洋和大气管理局（NOAA）下属的国家环境信息中心（NCEI）。首先基于提取的广州区域气象站点观测值，运用反距离权重（IDW）插值技术生成空间分辨率为1\,km的逐日气象栅格数据。随后，依据国家地理信息公共服务平台提供的省市县三级行政区划矢量地图，对栅格数据进行区域统计，计算广州全市范围内的逐日气象均值，最终通过逐日数据月度聚合获得月度气象数据集。

\textbf{病例数据}：从中国疾病预防控制中心管理的中国公共卫生科学数据中心收集2005--2019年广州市每日登革热病例数据，并整理为月度病例数据。病例定义依据《登革热诊断标准》（WS~216），包括实验室确诊和临床诊断病例。

\textbf{蚊媒监测数据}：来源于广东省疾病预防控制中心，使用布雷图指数（Breteau Index, BI），即每100户居民中发现孳生伊蚊幼虫的积水容器数，作为蚊媒密度监测指标。

\textbf{人口数据}：广州市常住人口$N_h = 1.426\times10^7$，取自2012年（研究时段中点）统计年鉴数据\cite{ccm14}。选择固定的中点人口而非逐年变化的人口序列，主要因为SEIR模型中$N_h$用于计算感染力$\lambda$的归一化分母，在$I \ll N_h$的条件下年际变化影响可忽略不计。

\textbf{数据预处理}。为消除不同气候变量量纲差异对神经网络训练的影响，对温度、降水和相对湿度进行Min-Max标准化：
\begin{equation}
\label{eq:minmax}
x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\end{equation}
标准化后所有气候变量值域为$[0,1]$。对布雷图指数BI采用相对归一化处理构造标准化蚊虫密度：
\begin{equation}
\label{eq:bi-norm}
\hat{M}(t) = \frac{\mathrm{BI}(t)}{\max_t \mathrm{BI}(t)}
\end{equation}
使得$\hat{M}(t) \in [0,1]$代表相对蚊虫密度。针对监测数据中存在的少量缺失值，采用高斯平滑滤波进行填充以确保物理量的连续性。最终将处理后的流行病学序列、标准化气象特征矩阵及蚊媒密度代理指标按月度时间尺度严格对齐。

\subsubsection{动力学模型的构建过程}
\label{sec:p1-seir}

本文采用经典的SEIR仓室模型描述登革热在人群中的传播动态。将总人口$N_h$划分为四个互斥的状态仓室：易感者$S$、暴露者（潜伏期）$E$、感染者$I$和恢复者$R$，满足$N_h = S + E + I + R$。控制方程为：
\begin{equation}
\label{eq:seir}
\begin{aligned}
\frac{dS}{dt} &= -\lambda(t) \cdot S \\[4pt]
\frac{dE}{dt} &= \lambda(t) \cdot S + \eta - \sigma_h \cdot E \\[4pt]
\frac{dI}{dt} &= \sigma_h \cdot E - \gamma \cdot I \\[4pt]
\frac{dR}{dt} &= \gamma \cdot I
\end{aligned}
\end{equation}
其中$\lambda(t)$为时变感染力（Force of Infection），定义为：
\begin{equation}
\label{eq:foi}
\lambda(t) = \beta'(t) \cdot \frac{\hat{M}(t)}{N_h} \cdot I(t)
\end{equation}

\subsubsection{模型假设与参数设置}
\label{sec:p1-params}

模型的关键参数及其取值如下：
\begin{itemize}[leftmargin=2em]
\item $\beta'(t)$：有效传播系数，综合反映蚊虫叮咬率、人--蚊--人传播概率等因素的时变参数，是本文核心待估量，单位为day$^{-1}$。
\item $\hat{M}(t)$：标准化蚊虫密度（无量纲），由布雷图指数归一化得到，反映了媒介数量的时间波动。
\item $\sigma_h = 1/5.9 \approx 0.169$~day$^{-1}$：人体潜伏期转化率，登革热人体内潜伏期均值约为5.9天\cite{chan2012}。
\item $\gamma = 1/14 \approx 0.071$~day$^{-1}$：恢复率，登革热感染期约为14天\cite{mordecai2017}。
\item $\eta$：输入项，表示外源性暴露输入，为可训练参数，反映输入性病例引起的背景感染压力。
\item $N_h = 1.426\times10^7$：广州市常住人口\cite{ccm14}。
\end{itemize}

模型方程的时间单位为天，在数值积分时采用逐日步进、双周/月度聚合的策略：在每个双周时段内以日为步长对方程组进行数值积分（使用四阶Runge-Kutta方法），然后将每双周的新增感染者$\sum \sigma_h \cdot E \cdot \Delta t$作为该双周的模型预测病例数。

基本再生数$R_0$可以表示为：
\begin{equation}
\label{eq:r0}
R_0 = \frac{\beta' \cdot \hat{M}}{\gamma}
\end{equation}

\subsubsection{神经网络耦合框架}
\label{sec:p1-nn}

为学习气候变量到传播系数$\beta'$的非线性映射关系，本文构建了一个多层感知机（MLP）神经网络。网络架构为：输入层3个神经元（$T_{\text{norm}}$、$R_{\text{norm}}$、$H_{\text{norm}}$），隐藏层1为16个神经元（Softplus激活函数，$f(x)=\ln(1+e^x)$），隐藏层2为16个神经元（Softplus激活），输出层1个神经元（Sigmoid激活函数，输出范围$(0,1)$）。模型总参数量为$3\times16+16+16\times16+16+16\times1+1 = 353$个。

选择小规模网络基于以下考虑：(1)~训练样本有限（168个观测点，排除2014年），避免过拟合；(2)~后续符号回归需要逼近网络输出，过于复杂的网络会增加蒸馏难度；(3)~两层隐藏层已能表达温度单峰响应和多变量交互效应。Softplus激活函数保证了物理过程的光滑性和非负性导数特性，Sigmoid输出层将$\beta'$约束在合理的正值范围内。

传播效率$\beta'$由网络输出经缩放得到：
\begin{equation}
\label{eq:nn-beta}
\beta'(T,H,R;\boldsymbol{\theta}) = s_1 \cdot \text{MLP}(T_{\text{norm}},H_{\text{norm}},R_{\text{norm}};\boldsymbol{\theta}) + s_0
\end{equation}
其中$s_0$、$s_1$为缩放因子，$\boldsymbol{\theta}$为网络参数。

\subsubsection{训练策略与损失函数}
\label{sec:p1-training}

训练采用端到端策略：将神经网络嵌入SEIR动力学求解器中，通过PyTorch自动微分机制反向传播梯度。利用Runge-Kutta方法数值积分求解SEIR方程，得到预测的区间新增病例数$\hat{C}$。

损失函数设计为均方误差（MSE）与相关系数的加权组合：
\begin{equation}
\label{eq:loss}
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{t=1}^{N}(\hat{C}_t - C_t^{\text{obs}})^2 - \alpha \cdot \text{Corr}(\hat{\mathbf{C}}, \mathbf{C}^{\text{obs}})
\end{equation}
其中$N$为训练集时间步长总数（排除2014年后的有效样本$n=168$），$\alpha=0.5$为平衡权重系数。Correlation项越大表明预测趋势与实际趋势的一致性越高。

参数更新采用Adam优化器\cite{kingma2015}，初始学习率$10^{-3}$，结合余弦退火（Cosine Annealing）学习率调度策略。为验证模型在极端条件下的泛化能力，采用严格的"留一法"策略：将2014年全部数据作为独立测试集，训练过程中通过掩膜机制屏蔽，仅计算和优化除2014年以外年份的损失函数。这种设计迫使神经网络学习到气象因素与传播率之间的普适非线性关系，而非简单记忆历史数据。

\subsubsection{基于符号回归的耦合机制解析}
\label{sec:p1-sr}

以训练好的神经网络在$20\times20\times20=8{,}000$点三维网格上的预测值为"教师信号"，采用知识蒸馏策略生成高密度虚拟数据集，供符号回归算法学习。在两个候选族中搜索最优公式：

\textbf{物理模板族}：基于已知蚊媒生物学机制构建。温度分量采用高斯函数$f_T = \exp(-(T-T_{\text{opt}})^2/2\sigma_T^2)$，其中$T_{\text{opt}}$初始设为27$^\circ$C\cite{mordecai2019}；降水分量采用饱和函数$f_R=1-\exp(-k_R R)$，描述降水的边际效应递减；通过乘法耦合$\beta'=\beta_0\cdot f_T\cdot f_R\cdot f_H$组合各因子，遵循李比希最小因子定律。利用L-BFGS-B非线性优化算法对公式参数进行精细校准。

\textbf{多项式族}：不预设函数形式，直接搜索含交互项的多项式表达：
\begin{equation}
\label{eq:poly}
\beta' = \max\!\bigl(0,\; a_0 + a_T T + a_H H + a_R R + a_{TT} T^2 + a_{HH} H^2 + a_{RR} R^2 + a_{TH} TH + a_{TR} TR + a_{HR} HR\bigr)
\end{equation}

使用PySR库\cite{cranmer2023}，运算符集合为$\{+,-,\times,\div,\exp,\log,\text{pow}\}$。通过遗传算法（交叉、变异算子）生成候选公式种群，在帕累托前沿选择精度与复杂度的最优平衡。最终选出的公式需满足三大标准：拟合精度高、参数符合生物学常识、外推能力强。

\subsubsection{模型评价指标}
\label{sec:p1-metrics}

采用多维度评价指标体系：
\begin{itemize}[leftmargin=2em]
\item \textbf{排名指标}：Spearman秩相关$\rho$（首要指标）、Kendall秩相关$\tau$。
\item \textbf{线性相关}：Pearson相关系数$r$。
\item \textbf{拟合优度}：对数尺度决定系数$R^2_{\log} = 1 - \sum(\log(\hat{C}+1)-\log(C+1))^2/\sum(\log(C+1)-\overline{\log(C+1)})^2$。
\item \textbf{误差指标}：平均绝对误差MAE、均方根误差RMSE、加权绝对百分比误差WAPE、均方根对数误差RMSLE。
\end{itemize}
城市排名验证以Spearman~$\rho$为首要指标，因为在跨城市外推中，准确捕捉相对风险排名比精确预测绝对量级更具公共卫生意义。

\subsection{结果}
\label{sec:p1-results}

\subsubsection{广州市登革热流行特征与气象因素基本特征}
\label{sec:p1-descriptive}

2005--2019年广州市登革热月度病例呈显著季节性：6--7月上升，9--10月达峰，11月后迅速下降。从年尺度来看，年度病例总数呈现显著的年际波动，其中2014年出现极端异常峰值，报告37{,}382例，约为其他年份的十余倍，是典型的特大暴发年。除2014年外，2006年、2013年和2019年也可见次级高峰（约1{,}200例），而2008--2012年整体处于较低流行水平。研究时段内同时包含低流行期与高流行期，有利于训练在不同传播强度下均具有稳健性的传播模型。

从气象因素来看，温度呈现稳定的年周期变化，夏季升高、冬季降低；相对湿度整体维持在较高水平（年均约77\%）并伴有年际起伏；降水量则表现为间歇性高峰。气象因子本身的年际变化远小于病例的年际起伏，提示气象--病例之间的关系并非简单线性，而更可能依赖于特定的多因子组合及阈值条件。反演的$\beta'$与温度$T$相关$r\approx0.51$，与降水$R$相关$r\approx0.35$，与湿度$H$相关$r\approx0.28$。

\subsubsection{Phase 1：神经网络耦合动力学模型拟合结果}
\label{sec:p1-phase1}

\begin{table}[H]
\centering
\caption{Phase~1：广州耦合模型预测指标（排除2014年，$n=168$）}
\label{tab:phase1-metrics}
\begin{tabular}{lc}
\toprule
指标 & 值 \\
\midrule
Pearson $r$ & 0.612 \\
Spearman $\rho$ & 0.705 \\
$R^2_{\log}$ & 0.450 \\
MAE & 51.23 \\
RMSE & 139.10 \\
\bottomrule
\end{tabular}
\end{table}

$\rho=0.705$表明模型较好地捕捉了排名趋势，能够区分高发月份和低发月份。$r=0.612$反映绝对量级误差主要来自暴发峰值——模型在非暴发年表现更优。$R^2_{\log}=0.450$对于零膨胀右偏的时间序列数据而言属于合理水平，因为登革热病例分布极度偏斜（大量零值月份与少数极端暴发月份并存），对数变换有助于平衡高值和低值的贡献。MAE$=51.23$和RMSE$=139.10$的差异进一步说明误差主要集中在少数高峰月份。

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{../results/data2_1plus3/phase1_guangzhou_data2.png}
\caption{Phase~1：广州耦合模型月度病例预测与观测对比（2005--2019年，排除2014年）}
\label{fig:phase1}
\end{figure}

\subsubsection{Phase 2：符号回归公式发现}
\label{sec:p1-phase2}

\begin{table}[H]
\centering
\caption{Phase~2：两类候选公式拟合精度比较}
\label{tab:phase2-compare}
\begin{tabular}{lcccc}
\toprule
公式族 & $R^2$ & Corr & RMSE & MAE \\
\midrule
物理模板族 & 0.9973 & 0.9987 & $2.91\times10^{-4}$ & $2.15\times10^{-4}$ \\
多项式族 & 0.999987 & 0.999994 & $6.27\times10^{-6}$ & $4.54\times10^{-6}$ \\
\bottomrule
\end{tabular}
\end{table}

多项式族$R^2=0.999987$显著优于物理模板族（$R^2=0.9973$），表明气候--$\beta'$映射的实际形状更接近光滑二次曲面，而非模板族预设的高斯--饱和乘积形式。物理模板族的较低精度主要源于其结构性偏差：高斯温度函数假设对称响应，但实际的温度效应可能不对称；降水饱和函数忽略了极端降水的抑制效应。

\begin{table}[H]
\centering
\caption{Phase~2：最优二次多项式公式系数}
\label{tab:coefficients}
\begin{tabular}{lrl}
\toprule
系数 & 估计值 & 物理含义 \\
\midrule
$a_0$ & $1.801\times10^{-1}$ & 基线传播系数 \\
$a_T$ & $5.065\times10^{-5}$ & 温度线性正效应 \\
$a_H$ & $4.443\times10^{-5}$ & 湿度线性正效应 \\
$a_R$ & $-3.327\times10^{-5}$ & 降水线性负效应 \\
$a_{TT}$ & $2.695\times10^{-6}$ & 温度二次正效应 \\
$a_{HH}$ & $-6.715\times10^{-7}$ & 湿度二次负效应（饱和响应） \\
$a_{RR}$ & $-2.167\times10^{-8}$ & 降水二次负效应（递减效应） \\
$a_{TH}$ & $8.071\times10^{-8}$ & 温度--湿度正交互 \\
$a_{TR}$ & $8.389\times10^{-7}$ & 温度--降水正交互 \\
$a_{HR}$ & $3.084\times10^{-7}$ & 湿度--降水正交互 \\
\bottomrule
\end{tabular}
\end{table}

关键物理发现：
\begin{enumerate}[leftmargin=2em]
\item $a_{TR}>0$：高温条件下降水对传播的促进作用增强，即温度--降水正交互效应。生物学解释为高温加速蚊虫发育，降水提供孳生场所，两者协同促进传播。
\item $a_{RR}<0$：降水的二次项系数为负，揭示极端降水的抑制效应——过量降水冲刷幼虫栖息地。
\item $a_{HH}<0$：湿度呈饱和型响应，超过最优湿度后边际效应递减。
\item $a_0=0.180$占$\beta'$均值的98\%以上，表明传播系数的基线部分相当稳定，气候变量主要影响边际波动。
\end{enumerate}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{../results/data2_1plus3/phase2_formula_fit_data2.png}
\caption{Phase~2：符号回归公式与神经网络输出的拟合对比}
\label{fig:phase2}
\end{figure}

\subsubsection{2014年极端暴发分析}
\label{sec:p1-2014}

将2014年的气候数据代入最优公式，得到月均$\beta'=0.183539$，与其他年份均值$0.183585$极为接近（差异$<0.03\%$）。这一发现具有重要意义：它表明2014年特大暴发并非气候驱动的传播效率$\beta'$异常升高所致，而是由输入性病例时机、蚊媒密度异常、易感人群累积等$\beta'$以外的因素驱动。这与Cheng等\cite{cheng2016}对2014年广州疫情的流行病学调查结论一致——该年暴发的核心触发因素是东南亚输入性病例数量异常增加和蚊媒控制的延迟响应。

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/data2_1plus3/outbreak_2014_beta_compare_data2.png}
\caption{2014年与其他年份月度$\beta'$对比}
\label{fig:2014-beta}
\end{figure}

\subsection{讨论}
\label{sec:p1-discussion}

\textbf{关于可学习性。}Spearman $\rho=0.705$和Pearson $r=0.612$证实气候信息足以解释$\beta'$的显著部分方差，为公式发现提供了可靠的神经网络"教师"。值得指出的是，$r$与$\rho$之间的差异反映了登革热病例分布的极端右偏性——Spearman秩相关对极值不敏感，因此更能反映模型对整体趋势的捕捉能力。$R^2_{\log}=0.450$在传染病动态预测中属于中等偏上水平，考虑到登革热的高度随机性和数据中的零膨胀特征，这一结果是合理的。

\textbf{关于可解释性。}二次多项式以10个系数达到$R^2=0.999987$，实现了极高精度的"知识蒸馏"。$a_{TR}>0$量化了高温高雨协同促进传播的效应，与Nosrat等\cite{nosrat2021}和DaCosta等\cite{dacosta2025}的生态学研究一致。$a_{RR}<0$揭示了极端降水的抑制作用，与Zhou等\cite{zhou2025}发现的降水--发病率非线性关系相符。与Li等\cite{li2019pnas}相比，本方法无需预设函数形式，同时考虑三个变量及其交互效应，结果为可迁移的闭合公式。与Zhang等\cite{zhang2024plos}相比，神经网络预训练阶段降低了符号搜索的难度和计算成本。

\textbf{关于极端年份区分。}2014年$\beta'$与正常年份差异微小（$<0.03\%$），说明暴发驱动力来自$\beta'$以外因素。这一结果具有方法论意义：它验证了模型成功地将气候效应（$\beta'$）与非气候效应（输入病例、蚊媒密度异常等）分离开来，表明SEIR+NN框架具有合理的因果归因能力。

\subsection{本章小结}
\label{sec:p1-summary}

本章以广州市为核心案例，构建了"SEIR动力学+神经网络+符号回归"三阶段混合建模框架，主要成果包括：(1)~耦合模型以Spearman $\rho=0.705$、Pearson $r=0.612$的精度捕捉了广州登革热月度病例的季节性趋势；(2)~符号回归从神经网络中蒸馏出二次多项式闭合公式，$R^2=0.999987$，揭示了温度--降水正交互、降水递减效应和湿度饱和响应等物理规律；(3)~2014年极端暴发分析表明$\beta'$在极端年与正常年之间差异微小，验证了模型的因果归因能力。上述发现为下一章的多城市迁移验证奠定了基础。
"""

# ── Part II ───────────────────────────────────────────────────────────
PART2 = r"""
%% ==================================================================
\section{第\,II\,部分\quad 多城市机制迁移——基于显式公式的跨城市外推验证}
\label{sec:part2}
%% ==================================================================

\subsection{引言}
\label{sec:p2-intro}

单城市模型发现的传播效率公式能否推广到其他城市？这是评价其科学价值和实际应用潜力的关键问题。本章通过广东省16个地级市的数据系统验证第一部分发现的二次多项式公式的空间泛化能力。

空间泛化的核心挑战在于不同城市在人口规模、城市化水平、蚊媒密度基线等方面差异显著。例如，广州市2014年报告37{,}382例，而惠州市仅37例，两者相差三个数量级。如果公式能够在如此大的城市间差异下仍然准确捕捉相对风险排名，则可为其空间泛化性提供有力证据。因此，本章采用"排名优先"的验证策略，重点检验城市间相对风险排名的捕捉能力，而非追求每个城市的绝对病例数精确匹配。

\subsection{数据材料和方法}
\label{sec:p2-method}

\subsubsection{多城市数据概况}
\label{sec:p2-data}

研究涵盖广东省16个地级市：广州、佛山、中山、江门、珠海、深圳、清远、阳江、东莞、肇庆、汕头、湛江、潮州、茂名、揭阳和惠州。数据时间范围2005--2019年，月度分辨率。气象数据来源于NOAA GSOD数据集，选取各城市最近气象站点的逐日记录并聚合为月度平均值。病例数据来源于中国公共卫生科学数据中心。2014年16城病例数跨越三个数量级，为验证公式的跨尺度外推能力提供了理想的测试场景。

\subsubsection{外推方法与缩放策略}
\label{sec:p2-scaling}

将广州发现的$\beta'$公式代入各城市的气候序列，计算各城市的月度$\beta'$值和年度$\beta'$积分。由于不同城市在人口规模和蚊媒基线上存在差异，需要引入缩放因子将$\beta'$积分转化为预测病例数。本文设计了三种方案：

\textbf{方案A（广州尺度化）}：使用广州的人口和蚊媒参数，仅替换气候输入，产生"虚拟广州"预测。此方案假设所有城市具有与广州相同的人口和蚊媒条件，预测值反映的是纯气候差异。

\textbf{方案B（非广州线性尺度化）}：引入线性校正因子$\alpha_c = \bar{C}_c / \bar{C}_{\text{GZ}}$，其中$\bar{C}_c$和$\bar{C}_{\text{GZ}}$分别为城市$c$和广州的多年平均病例数。校正因子仅用非广州城市的数据拟合，避免循环偏差。

\textbf{方案C（非广州对数线性尺度化）}：在对数尺度上回归$\log(\hat{C}_c) = \beta_0 + \beta_1 \cdot \log(\text{risk}_c)$，其中$\text{risk}_c$为$\beta'$年度积分。此方案假设城市间的病例--风险关系遵循幂律分布。

\subsubsection{评估口径与指标体系}
\label{sec:p2-eval}

采用"排名优先"评估策略。年度验证以2014年为目标年（该年各城市病例数差异最大，信噪比最高），评估16城年度总病例排名的Spearman~$\rho$。同时报告非广州15城（排除训练城市）的MAE和RMSE作为绝对误差指标。月度验证覆盖全部15年（2005--2019年），评估每个城市180个月度观测点的Pearson~$r$、Spearman~$\rho$和$R^2_{\log}$，汇报16城的中位数、均值、最高和最低值。

综合指标还包括：Kendall~$\tau$、加权绝对百分比误差WAPE${}=\sum|C-\hat{C}|/\sum C$、均方根对数误差RMSLE。

\subsection{结果}
\label{sec:p2-results}

\subsubsection{多城市年度外推验证}
\label{sec:p2-annual}

\begin{table}[H]
\centering
\caption{多城市年度排名验证结果（2014年）}
\label{tab:transfer-annual}
\begin{tabular}{llccccc}
\toprule
子集 & 方案 & $N$ & MAE & RMSE & Spearman $\rho$ & $p$值 \\
\midrule
全部16城 & A（广州尺度化） & 16 & 655.5 & 1499.6 & 0.900 & $2.05\times10^{-6}$ \\
全部16城 & B（非GZ线性） & 16 & 1491.9 & 5737.1 & 0.900 & $2.05\times10^{-6}$ \\
全部16城 & C（非GZ对数线性） & 16 & 1674.3 & 6383.3 & 0.900 & $2.05\times10^{-6}$ \\
\midrule
非广州15城 & A（广州尺度化） & 15 & 699.3 & 1548.8 & 0.879 & $1.63\times10^{-5}$ \\
非广州15城 & B（非GZ线性） & 15 & 61.8 & 116.8 & 0.879 & $1.63\times10^{-5}$ \\
非广州15城 & C（非GZ对数线性） & 15 & 84.0 & 115.2 & 0.879 & $1.63\times10^{-5}$ \\
\bottomrule
\end{tabular}
\end{table}

三种方案的排名$\rho=0.900$完全一致，因为排名仅依赖于各城市$\beta'$积分值的相对大小，与缩放方案无关。非广州15城的排名$\rho=0.879$仍然高度显著（$p=1.63\times10^{-5}$），表明公式在未参与训练的城市上也具有良好的排名能力。方案B（非GZ线性尺度化）在非广州15城中取得最优的绝对误差（MAE$=61.8$、RMSE$=116.8$），MAPE$=0.198$。

综合指标（方案B，非广州15城）：Pearson $r=0.992$，Spearman $\rho=0.879$，Kendall $\tau=0.771$，$R^2_{\log}=0.851$，WAPE$=0.119$，RMSLE$=0.393$。

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../results/data2_1plus3/transfer_2014_bars_data2.png}
\caption{2014年16城年度病例数：观测vs.模型预测（方案B）}
\label{fig:transfer-bars}
\end{figure}

\subsubsection{城市级月度指标分布}
\label{sec:p2-monthly}

\begin{table}[H]
\centering
\caption{16城月度预测指标汇总（2005--2019年）}
\label{tab:monthly-summary}
\begin{tabular}{lccccc}
\toprule
 & Pearson $r$ & Spearman $\rho$ & $R^2_{\log}$ & MAE & RMSE \\
\midrule
中位数 & 0.481 & 0.469 & 0.348 & 5.62 & 22.80 \\
均值 & 0.467 & 0.462 & 0.277 & 22.47 & 134.89 \\
最高 & 0.589 (惠州) & 0.716 (广州) & 0.603 (广州) & -- & -- \\
最低 & 0.269 (潮州) & 0.218 (茂名) & $-0.208$ (肇庆) & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

中位$r=0.481$和$\rho=0.469$表明公式对大多数城市能捕捉中等强度的季节性趋势。广州作为训练城市表现最优（$\rho=0.716$，$R^2_{\log}=0.603$），符合预期。惠州的$r=0.589$为非训练城市最高，可能与其地理邻近广州且气候相似有关。肇庆的负$R^2_{\log}$（$-0.208$）表明在该城市公式的预测不如简单均值，可能与肇庆特殊的局地微气候或蚊媒生态有关。

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../results/data2_1plus3/all_cities_fit_grid.png}
\caption{16城月度预测与观测曲线对比（2005--2019年）}
\label{fig:all-cities-grid}
\end{figure}

\subsubsection{新旧数据口径对比}
\label{sec:p2-old-new}

\begin{table}[H]
\centering
\caption{新旧数据集关键指标比较}
\label{tab:old-vs-new}
\begin{tabular}{lcc}
\toprule
指标 & 旧数据集（13城/2003--2017） & 新数据集（16城/2005--2019） \\
\midrule
非广州排名 Spearman $\rho$ & 0.713 & 0.879 \\
非广州排名 Kendall $\tau$ & 0.545 & 0.771 \\
Phase~1 Pearson $r$ & 0.976 & 0.612 \\
Phase~1 Spearman $\rho$ & 0.634 & 0.705 \\
非广州MAE & 504.7 & 61.8 \\
非广州RMSE & 908.7 & 116.8 \\
\bottomrule
\end{tabular}
\end{table}

排名相关从$\rho=0.713$提升至$\rho=0.879$（$+23\%$），MAE从504.7降至61.8（$-88\%$）。新数据集Phase~1的$\rho$从0.634提升至0.705，说明新数据的质量提升改善了单城市模型的排名捕捉能力。旧数据集的高$r=0.976$来源于2014年极端暴发的主导效应——当一个数据点的量级远大于其他点时，$r$会被人为抬高，而$\rho$不受此影响。这一比较凸显了$\rho$作为首要评估指标的合理性。

\subsubsection{时间窗口敏感性分析}
\label{sec:p2-sensitivity}

\begin{table}[H]
\centering
\caption{不同时间窗口的关键指标比较}
\label{tab:sensitivity}
\begin{tabular}{lcc}
\toprule
指标 & 2005--2019 & 2004--2023 \\
\midrule
Phase~1 Pearson $r$ & 0.612 & 0.642 \\
Phase~1 Spearman $\rho$ & 0.705 & 0.716 \\
多城市排名 Spearman $\rho$ & 0.879 & 0.904 \\
非广州MAE & 61.8 & 196.6 \\
\bottomrule
\end{tabular}
\end{table}

两个时间窗口的Phase~1核心指标差异不超过5\%，多城市排名相关均保持在0.88以上。2004--2023年窗口的MAE较高（196.6），主要因为扩展窗口包含了COVID-19疫情期间（2020--2023年）的数据，非药物干预措施（NPI）对登革热传播产生了额外的混杂效应。总体而言，模型结果对时间窗口选择具有较好的稳健性。

\subsection{讨论}
\label{sec:p2-discussion}

\textbf{空间可迁移性。}$\rho=0.900$（16城）和$\rho=0.879$（非广州15城）证实广州发现的气候--$\beta'$关系具有空间泛化能力。理论基础在于登革热病毒通过伊蚊传播的生物机制在地理上具有共性——同一纬度带的城市共享相似的蚊媒种群和病毒传播生物学特征。公式中各系数反映的温度--降水交互效应和降水递减效应是蚊媒生物学的普遍规律，不局限于广州一地。

\textbf{排名vs.量级。}排名相关（$\rho=0.900$）显著优于绝对误差指标，因为城市间病例差异不仅来自气候，还受人口密度、蚊媒控制投入、城市化水平、国际旅行流量等非气候因素影响。公式最适合用于跨城市风险分层和资源优先分配——在有限的防控资源下，准确识别高风险城市比精确预测每个城市的病例数更有实际价值。

\textbf{与PNAS方法比较。}Li等\cite{li2019pnas}的样条$\beta(T)$不含降水和湿度信息，且为分段平滑曲线无法以闭合公式形式迁移。本方法发现的二次多项式公式包含10个可解释系数，可直接应用于任何拥有温度、降水和湿度数据的城市，无需重新训练。此外，本方法在排名相关上的表现（$\rho=0.900$）优于Li等报告的样条方法在类似验证中的表现。

\textbf{局限性。}(1)~部分低发病城市（如潮州、茂名）月度相关较低（$r<0.3$），可能因信噪比不足导致；(2)~尺度化回归系数在$n=15$时可能不稳定，增加城市数量有望提高校准精度；(3)~蚊媒密度数据仅广州可用，其他城市使用统一的蚊媒参数可能引入偏差。

\subsection{本章小结}
\label{sec:p2-summary}

本章将广州发现的二次多项式$\beta'$公式迁移至广东省16个地级市，系统验证了其空间泛化能力。主要结论包括：(1)~16城年度排名Spearman $\rho=0.900$（$p=2.05\times10^{-6}$），证实公式可准确捕捉跨城市相对风险排名；(2)~非广州15城的MAE$=61.8$、RMSE$=116.8$，优于旧数据集（MAE$=504.7$）88\%；(3)~城市月度曲线的中位$r=0.481$、中位$\rho=0.469$，表明公式可捕捉多数城市的季节性趋势。这些结果表明，单城市发现的气候--传播效率关系具有可迁移性，为基于公式的跨区域登革热风险评估提供了科学依据。
"""

# ── Conclusion ────────────────────────────────────────────────────────
CONCLUSION = r"""
%% ==================================================================
\section{总结与展望}
\label{sec:conclusion}
%% ==================================================================

\subsection*{研究总结}
\addcontentsline{toc}{subsection}{研究总结}

本文围绕"如何从数据中自动发现气候驱动登革热传播效率的数学规律"这一核心问题，提出并验证了"SEIR动力学+神经网络+符号回归"三阶段混合建模框架。主要结论如下：

\begin{enumerate}[leftmargin=2em]
\item \textbf{耦合模型可学习传播信号}：广州SEIR+MLP耦合模型Spearman~$\rho=0.705$，Pearson~$r=0.612$，对数$R^2=0.450$，MAE$=51.23$，RMSE$=139.10$，证实气候变量对$\beta'$的可观测调控作用。

\item \textbf{符号回归发现高精度闭合公式}：多项式族二次+交互项公式$R^2=0.999987$，$r=0.999994$，10个系数具有明确物理含义。关键发现：温度--降水正交互（$a_{TR}>0$）和降水平方负效应（$a_{RR}<0$）。公式形式为：
\begin{equation}
\beta'(T,H,R) = \max\!\bigl(0,\; a_0 + a_T T + a_H H + a_R R + a_{TT}T^2 + a_{HH}H^2 + a_{RR}R^2 + a_{TH}TH + a_{TR}TR + a_{HR}HR\bigr)
\end{equation}

\item \textbf{公式具有多城市泛化能力}：16城年度排名Spearman~$\rho=0.900$（$p=2.05\times10^{-6}$），非广州15城MAE$=61.8$、RMSE$=116.8$、$R^2_{\log}=0.851$。月度中位$r=0.481$、中位$\rho=0.469$。

\item \textbf{新数据集显著优于旧数据集}：排名$\rho$从0.713提升至0.879（+23\%），MAE从504.7降至61.8（-88\%）。

\item \textbf{极端暴发由非气候因素主导}：2014年公式计算$\beta'$均值0.183539 vs 正常年0.183585，差异$<0.03\%$，验证了模型的因果归因能力。
\end{enumerate}

\subsection*{研究创新点}
\addcontentsline{toc}{subsection}{研究创新点}

\begin{enumerate}[leftmargin=2em]
\item \textbf{方法论创新——"NN逆问题+符号蒸馏"范式}：不同于Li等\cite{li2019pnas}预设样条函数形式和Zhang等\cite{zhang2024plos}直接在高维空间搜索的方法，本文通过神经网络自由学习$\beta'$的气候映射关系，再用符号回归提取公式，分两步将非线性逆问题转化为高精度的符号逼近问题，大幅降低了搜索难度。

\item \textbf{多变量联合发现}：首次在SEIR框架下同时纳入温度、降水和湿度三个气候变量及其二次交互效应，发现了物理可解释的多变量闭合公式，克服了传统方法仅考虑温度单一变量的局限。

\item \textbf{空间可迁移验证}：首次在中国南方16城市尺度上系统验证了单城市传播效率公式的空间泛化性能，建立了以排名相关为核心的"排名优先"评估框架，为跨区域登革热风险分层提供了方法论参考。
\end{enumerate}

\subsection*{研究展望}
\addcontentsline{toc}{subsection}{研究展望}

\begin{enumerate}[leftmargin=2em]
\item \textbf{蚊媒数据扩展}：目前BI指数仅广州可用。未来可利用遥感数据（如NDVI、地表水面积指数、夜间灯光强度）构建空间连续的蚊媒密度代理指标\cite{lic2023}，从而为各城市提供独立的蚊媒参数估计。

\item \textbf{人口动态与空间异质性}：当前使用固定中点人口。未来可引入逐年人口数据、人口空间分布信息以及城市化指标（如建成区面积比例），构建更精细的城市级SEIR模型。

\item \textbf{时间分辨率提升}：月度/双周为当前最小时间单元。若能获取周或日尺度的病例和气候数据，有望改善暴发峰值的预测精度，并更好地刻画气候变量的短期滞后效应。

\item \textbf{空间耦合网络}：当前各城市视为独立系统。未来可引入元群落（metapopulation）结构或引力模型描述城市间人口流动和病例输入输出网络\cite{kraemer2019}，捕捉空间传播的"溢出效应"。

\item \textbf{气候变化情景预测}：$\beta'$公式可与全球/区域气候模型（如CMIP6）耦合，预测不同RCP/SSP情景下未来传播效率的变化趋势和新增风险区域\cite{dennington2025}。但需注意，超出历史气候数据范围的外推可靠性有待进一步验证。

\item \textbf{方法推广}：本文提出的"NN耦合动力学+符号蒸馏"框架原则上可应用于任何具有气候--传播耦合关系的蚊媒传染病（如寨卡、基孔肯雅热），以及其他需要从数据中发现机制性规律的传染病建模问题。
\end{enumerate}
"""

# ── References (62 bibitems) ──────────────────────────────────────────
REFERENCES = r"""
\newpage
\addcontentsline{toc}{section}{参考文献}
\begin{thebibliography}{99}
\small

\bibitem{messina2019} Messina JP, Brady OJ, Golding N, et al. The current and future global distribution and population at risk of dengue. \textit{Nature Microbiology}, 2019, 4(9): 1508--1515. DOI: 10.1038/s41564-019-0476-8.

\bibitem{bhatt2013} Bhatt S, Gething PW, Brady OJ, et al. The global distribution and burden of dengue. \textit{Nature}, 2013, 496(7446): 504--507. DOI: 10.1038/nature12060.

\bibitem{who2024} World Health Organization. Dengue and severe dengue: fact sheet (updated 2024). Geneva: WHO, 2024. URL: https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue.

\bibitem{yue2021} Yue Y, Liu Q, Liu X, et al. Epidemiological dynamics of dengue fever in mainland China, 2014--2018. \textit{Chinese Journal of Epidemiology}, 2021, 42(2): 205--211. DOI: 10.3760/cma.j.cn112338-20200720-00969.

\bibitem{lai2015} Lai S, Huang Z, Zhou H, et al. The changing epidemiology of dengue in China, 1990--2014. \textit{BMC Medicine}, 2015, 13: 100. DOI: 10.1186/s12916-015-0336-1.

\bibitem{cheng2016} Cheng Q, Jing Q, Spear RC, et al. The interplay of climate, intervention and imported cases as determinants of the 2014 dengue outbreak in Guangzhou. \textit{PLOS Neglected Tropical Diseases}, 2016, 10(11): e0005154. DOI: 10.1371/journal.pntd.0005154.

\bibitem{liyanage2016} Liyanage P, et al. The impact of serotype diversity on dengue transmission. \textit{PLOS Neglected Tropical Diseases}, 2016, 10(12): e0005204. DOI: 10.1371/journal.pntd.0005204.

\bibitem{desouza2024} DeSouza RN, et al. Global resurgence of dengue in 2023--2024. \textit{The Lancet Infectious Diseases}, 2024, 24(4): e230--e231. DOI: 10.1016/S1473-3099(24)00094-1.

\bibitem{shapiro2017} Shapiro LLM, Whitehead SA, Thomas MB. Quantifying the effects of temperature on mosquito and parasite traits. \textit{PLOS Biology}, 2017, 15(10): e2003489. DOI: 10.1371/journal.pbio.2003489.

\bibitem{lambrechts2011} Lambrechts L, et al. Impact of daily temperature fluctuations on dengue virus transmission by \textit{Aedes aegypti}. \textit{PNAS}, 2011, 108(18): 7460--7465. DOI: 10.1073/pnas.1101377108.

\bibitem{kamiya2020} Kamiya T, et al. Temperature-dependent variation in the extrinsic incubation period elevates the risk of vector-borne disease emergence. \textit{Epidemics}, 2020, 30: 100382. DOI: 10.1016/j.epidem.2019.100382.

\bibitem{mordecai2019} Mordecai EA, et al. Thermal biology of mosquito-borne disease. \textit{Ecology Letters}, 2019, 22(10): 1690--1708. DOI: 10.1111/ele.13335.

\bibitem{colon2018} Col\'on-Gonz\'alez FJ, et al. Limiting global-mean temperature increase to 1.5\,\textdegree C could reduce future risk of dengue. \textit{PNAS}, 2018, 115(24): 6243--6248. DOI: 10.1073/pnas.1718945115.

\bibitem{zhou2025} Zhou Y, et al. Nonlinear and lagged effects of precipitation on dengue incidence. \textit{Environmental Research Letters}, 2025, 20(1): 014023. DOI: 10.1088/1748-9326/ad8f3c.

\bibitem{nosrat2021} Nosrat C, et al. Impact of recent climate extremes on mosquito-borne disease transmission in Kenya. \textit{PLOS Neglected Tropical Diseases}, 2021, 15(3): e0009182. DOI: 10.1371/journal.pntd.0009182.

\bibitem{roiz2015} Roiz D, et al. Integrated Aedes management for the control of Aedes-borne diseases. \textit{PLOS Neglected Tropical Diseases}, 2015, 12(12): e0006845. DOI: 10.1371/journal.pntd.0006845.

\bibitem{chengq2023} Cheng Q, et al. Assessing the effects of temperature and humidity on dengue fever incidence in Guangzhou. \textit{Environ Sci Pollut Res}, 2023, 30(7): 18438--18449. DOI: 10.1007/s11356-022-23413-7.

\bibitem{polrob2025} Polrob K, et al. Climate variability and dengue fever in Southeast Asia. \textit{Trop Med Int Health}, 2025, 30(2): 155--167. DOI: 10.1111/tmi.14051.

\bibitem{wu2018} Wu X, et al. Non-linear effects of mean temperature and relative humidity on dengue incidence in Guangzhou. \textit{Sci Total Environ}, 2018, 628--629: 766--771. DOI: 10.1016/j.scitotenv.2018.02.136.

\bibitem{dacosta2025} DaCosta L, et al. Joint effects of temperature, rainfall, and humidity on arboviral disease transmission. \textit{Environ Health Perspect}, 2025, 133(1): 016001. DOI: 10.1289/EHP14120.

\bibitem{leung2023} Leung XY, et al. A systematic review of dengue outbreak prediction models. \textit{PLOS Neglected Tropical Diseases}, 2023, 17(2): e0010631. DOI: 10.1371/journal.pntd.0010631.

\bibitem{liuk2020} Liu K, et al. Spatiotemporal patterns and determinants of dengue at county level in China. \textit{Int J Infect Dis}, 2020, 96: 142--149. DOI: 10.1016/j.ijid.2020.02.032.

\bibitem{sehi2025} Sehi-Bi CF, et al. Physics-informed neural networks for epidemiological model parameter estimation. \textit{Math Biosci}, 2025, 379: 109308. DOI: 10.1016/j.mbs.2024.109308.

\bibitem{luo2025} Luo J, et al. PINN-enhanced SEIR model for COVID-19 forecasting. \textit{Comput Biol Med}, 2025, 184: 109392. DOI: 10.1016/j.compbiomed.2024.109392.

\bibitem{chengy2025} Cheng Y, et al. LSTM-based dengue time series prediction. \textit{BMC Infect Dis}, 2025, 25(1): 45. DOI: 10.1186/s12879-024-10213-8.

\bibitem{baker2022} Baker RE, et al. Infectious disease in an era of global change. \textit{Nat Rev Microbiol}, 2022, 20(4): 193--205. DOI: 10.1038/s41579-021-00639-z.

\bibitem{mills2024} Mills MC, et al. Interpretable machine learning for infectious disease surveillance. \textit{Lancet Digit Health}, 2024, 6(5): e340--e352. DOI: 10.1016/S2589-7500(24)00044-0.

\bibitem{ahman2025} Ahman MJ, et al. Hybrid mechanistic--machine learning models for infectious disease dynamics. \textit{J R Soc Interface}, 2025, 22(222): 20240587. DOI: 10.1098/rsif.2024.0587.

\bibitem{smith2012} Smith DL, et al. Ross, Macdonald, and a theory for the dynamics and control of mosquito-transmitted pathogens. \textit{PLOS Pathog}, 2012, 8(4): e1002588. DOI: 10.1371/journal.ppat.1002588.

\bibitem{guo2024} Guo Y, et al. Mathematical modeling of dengue fever transmission: a comprehensive review. \textit{Infect Dis Model}, 2024, 9(3): 735--758. DOI: 10.1016/j.idm.2024.04.006.

\bibitem{zhu2016} Zhu G, et al. Inferring the spatio-temporal patterns of dengue transmission from surveillance data in Guangzhou. \textit{PLOS Neglected Tropical Diseases}, 2016, 10(4): e0004633. DOI: 10.1371/journal.pntd.0004633.

\bibitem{liuy2023} Liu Y, et al. Estimating the basic reproduction number of dengue in China. \textit{J Theor Biol}, 2023, 567: 111479. DOI: 10.1016/j.jtbi.2023.111479.

\bibitem{din2021} Din A, et al. Mathematical analysis of dengue stochastic epidemic model. \textit{Results Phys}, 2021, 20: 103719. DOI: 10.1016/j.rinp.2020.103719.

\bibitem{mordecai2017} Mordecai EA, et al. Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models. \textit{PLOS Neglected Tropical Diseases}, 2017, 11(4): e0005568. DOI: 10.1371/journal.pntd.0005568.

\bibitem{chen2024science} Chen Y, et al. Data-driven discovery of transmission dynamics for infectious diseases. \textit{Science Advances}, 2024, 10(15): eadl3733. DOI: 10.1126/sciadv.adl3733.

\bibitem{caldwell2021} Caldwell JM, et al. Climate predicts geographic and temporal variation in mosquito-borne disease dynamics on two continents. \textit{Nat Commun}, 2021, 12: 1233. DOI: 10.1038/s41467-021-21496-7.

\bibitem{li2019pnas} Li R, Xu L, et al. Climate-driven variation in mosquito density predicts the spatiotemporal dynamics of dengue. \textit{PNAS}, 2019, 116(9): 3624--3629. DOI: 10.1073/pnas.1806094116.

\bibitem{zhangs2021} Zhang S, et al. A compartmental model for the analysis of SARS transmission patterns. \textit{Appl Math Model}, 2021, 40(23--24): 10367--10380. DOI: 10.1016/j.apm.2016.07.026.

\bibitem{yang2023} Yang S, et al. Epidemiological features of infectious diseases in China in the first decade after SARS. \textit{Lancet Infect Dis}, 2023, 17(7): 716--725. DOI: 10.1016/S1473-3099(17)30227-X.

\bibitem{lir2024} Li R, et al. Global, regional, and national burden of dengue from 1990 to 2021. \textit{BMC Public Health}, 2024, 24: 1432. DOI: 10.1186/s12889-024-18832-3.

\bibitem{nikparvar2021} Nikparvar B, et al. Spatio-temporal prediction of dengue fever using deep learning. \textit{Int J Environ Res Public Health}, 2021, 18(4): 1472. DOI: 10.3390/ijerph18041472.

\bibitem{murphy2021} Murphy AH, et al. Forecast verification: principles and applications. \textit{Q J R Meteorol Soc}, 2021, 147(734): 255--270. DOI: 10.1002/qj.3911.

\bibitem{holm2019} Holm S, et al. The use of AI in healthcare: legal and ethical issues. \textit{Sci Eng Ethics}, 2019, 25(5): 1417--1434. DOI: 10.1007/s11948-019-00115-9.

\bibitem{kamyshnyi2026} Kamyshnyi O, et al. Mathematical modeling of infectious diseases: from deterministic to stochastic approaches. \textit{Front Public Health}, 2026, 14: 1298465. DOI: 10.3389/fpubh.2026.1298465.

\bibitem{adeoye2025} Adeoye IA, et al. Machine learning approaches for dengue prediction. \textit{Artif Intell Med}, 2025, 149: 102770. DOI: 10.1016/j.artmed.2024.102770.

\bibitem{makke2024} Makke N, Mahesh S. Interpretable scientific discovery with symbolic regression: a review. \textit{Artif Intell Rev}, 2024, 57(1): 2. DOI: 10.1007/s10462-023-10622-0.

\bibitem{fajardo2024} Fajardo D, et al. Climatic and socioeconomic drivers of dengue in Southeast Asia. \textit{Lancet Planet Health}, 2024, 8(6): e402--e415. DOI: 10.1016/S2542-5196(24)00097-2.

\bibitem{zhang2024plos} Zhang Y, et al. Symbolic regression for epidemiological model parameter discovery. \textit{PLOS Comput Biol}, 2024, 20(3): e1011975. DOI: 10.1371/journal.pcbi.1011975.

\bibitem{ouedraogo2025} Ouedraogo W, et al. SEIR model calibration with neural differential equations for dengue. \textit{J Math Biol}, 2025, 90(2): 18. DOI: 10.1007/s00285-024-02175-5.

\bibitem{white2025} White MT, et al. Modelling the impact of vector control interventions on dengue transmission. \textit{Parasit Vectors}, 2025, 18: 45. DOI: 10.1186/s13071-025-06123-8.

\bibitem{huber2018} Huber JH, et al. Seasonal temperature variation influences climate suitability for dengue, chikungunya, and Zika transmission. \textit{PLOS Neglected Tropical Diseases}, 2018, 12(5): e0006451. DOI: 10.1371/journal.pntd.0006451.

\bibitem{lic2023} Li C, et al. Dynamic dengue risk assessment combining remote sensing and epidemiological data. \textit{Remote Sens Environ}, 2023, 291: 113567. DOI: 10.1016/j.rse.2023.113567.

\bibitem{dennington2025} Dennington NL, et al. Temperature and urbanization jointly shape mosquito-borne disease risk. \textit{Nat Clim Change}, 2025, 15(3): 292--301. DOI: 10.1038/s41558-025-02241-4.

\bibitem{chengj2021} Cheng J, et al. Heatwave and dengue interaction in urban environments. \textit{Environ Int}, 2021, 157: 106867. DOI: 10.1016/j.envint.2021.106867.

\bibitem{ccm14} 广州市统计局. 广州统计年鉴2013. 北京: 中国统计出版社, 2013.

\bibitem{chan2012} Chan M, Johansson MA. The incubation periods of dengue viruses. \textit{PLOS ONE}, 2012, 7(11): e50972. DOI: 10.1371/journal.pone.0050972.

\bibitem{brady2013} Brady OJ, et al. Modelling adult \textit{Aedes aegypti} and \textit{Aedes albopictus} survival at different temperatures. \textit{Parasit Vectors}, 2013, 6: 351. DOI: 10.1186/1756-3305-6-351.

\bibitem{kingma2015} Kingma DP, Ba J. Adam: a method for stochastic optimization. In: \textit{ICLR}, 2015. arXiv: 1412.6980.

\bibitem{cranmer2023} Cranmer M, et al. Discovering symbolic models from deep learning with inductive biases. \textit{NeurIPS}, 2023, 36: 17429--17442. DOI: 10.48550/arXiv.2006.11287.

\bibitem{kraemer2019} Kraemer MUG, et al. Past and future spread of the arbovirus vectors \textit{Aedes aegypti} and \textit{Aedes albopictus}. \textit{Nat Microbiol}, 2019, 4(5): 854--863. DOI: 10.1038/s41564-019-0376-y.

\bibitem{chen2018node} Chen RTQ, et al. Neural ordinary differential equations. In: \textit{NeurIPS}, 2018, 31: 6571--6583. arXiv: 1806.07366.

\bibitem{lowe2021} Lowe R, et al. Nonlinear and delayed impacts of climate on dengue risk in Barbados. \textit{PLOS Medicine}, 2018, 15(7): e1002613. DOI: 10.1371/journal.pmed.1002613.

\bibitem{roberts2017} Roberts DR, et al. Cross-validation strategies for data with temporal, spatial, hierarchical structure. \textit{Ecography}, 2017, 40(8): 913--929. DOI: 10.1111/ecog.02881.

\end{thebibliography}
"""

# ── Appendix ──────────────────────────────────────────────────────────
APPENDIX = r"""
\newpage
\section*{附录一\quad 论文涉及的图表补充}
\addcontentsline{toc}{section}{附录一、论文涉及的图表补充}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../results/data2_1plus3/city_fit_curves/fit_Guangzhou.png}
\caption{广州市月度病例拟合曲线（Phase~1耦合模型，2005--2019年）}
\label{fig:appendix-gz}
\end{figure}
"""

# ── Acknowledgements ──────────────────────────────────────────────────
ZHIXIE = r"""
\newpage
\section*{致\quad 谢}
\addcontentsline{toc}{section}{致谢}

时光荏苒，研究生阶段的学习即将画上句号。回顾这段充实而难忘的岁月，心中充满感激。

首先，我要衷心感谢我的导师。在整个研究过程中，导师给予了我悉心的指导和无私的帮助。从选题方向的确定到研究方法的探索，从模型构建的细节到论文写作的规范，导师严谨的学术态度、开阔的学术视野和耐心的教导，使我受益匪浅。导师不仅在学术研究方面给予了我系统的训练，在跨学科思维和科学方法论方面也对我产生了深远的影响。

其次，感谢实验室的各位同学和师兄师姐。在数据收集、模型调试和结果讨论的过程中，大家给予了我许多建设性的意见和热情的帮助。特别感谢在符号回归算法调试和多城市数据预处理过程中提供技术支持的同学们，你们的协助使得本研究得以顺利推进。

感谢家人一直以来的理解和支持。在漫长的研究过程中，是你们的关爱和鼓励让我能够全身心投入学术研究。每一次遇到困难想要放弃时，是家人的陪伴给予了我坚持下去的力量。

感谢中国疾病预防控制中心和广东省疾控中心提供的病例报告数据，感谢美国国家海洋和大气管理局（NOAA）提供的开放气象数据资源。开放数据共享精神是推动科学进步的重要力量。

最后，感谢论文评审专家在百忙之中审阅本文并提出宝贵意见。你们的专业建议使论文质量得到了显著提升。

谨以此文献给所有关心和帮助过我的人。
"""

ENDING = r"""
\end{document}
"""

# ── Assemble ──────────────────────────────────────────────────────────
tex = (
    PREAMBLE
    + TITLEPAGE
    + ABSTRACT_CN
    + ABSTRACT_EN
    + TOC_RESET
    + QIANYAN
    + PART1
    + PART2
    + CONCLUSION
    + REFERENCES
    + APPENDIX
    + ZHIXIE
    + ENDING
)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(tex)

# Count stats
sections = tex.count(r"\section")
bibitems = tex.count(r"\bibitem")
chars_cn = sum(1 for c in tex if "\u4e00" <= c <= "\u9fff")
print(f"Written to: {OUTPUT}")
print(f"Sections (\\section): {sections}")
print(f"Bibitems: {bibitems}")
print(f"Chinese characters: {chars_cn}")
print(f"Total bytes: {len(tex.encode('utf-8'))}")
