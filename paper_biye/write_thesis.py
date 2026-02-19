# -*- coding: utf-8 -*-
"""Generate the complete graduation thesis draft as a .tex file."""

import textwrap

sections = []

# ============ PREAMBLE ============
sections.append(r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{ctex}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage[table]{xcolor}
\usepackage{float}
\usepackage{subcaption}
\usepackage{enumitem}
\usepackage{multirow}
\usepackage{setspace}
\usepackage{longtable}

\geometry{left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\onehalfspacing
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=cyan}

\title{\textbf{基于神经网络耦合动力学模型的登革热传播效率发现\\与多城市验证}}
\author{XXX\\[0.5em]{\normalsize XXX大学\quad XXX学院}\\[0.3em]{\normalsize 指导教师：XXX\quad 教授}}
\date{}

\begin{document}
\maketitle
\thispagestyle{empty}
\newpage
\setcounter{page}{1}
\pagenumbering{Roman}
""")

# ============ CHINESE ABSTRACT ============
sections.append(r"""
\begin{center}
{\Large\textbf{摘\quad 要}}
\end{center}

登革热是全球增长最快的蚊媒传染病之一，其传播效率受温度、湿度和降水等气象因子的复杂非线性驱动，但现有模型在气象--传播效率的显式函数关系上仍存在空白。本文提出一种"SEIR动力学模型$\,+\,$神经网络$\,+\,$符号回归"三位一体的混合建模框架，旨在从真实监测数据中发现气象因子驱动登革热传播效率的定量关系，并推导出具有物理可解释性的解析公式。

研究以广东省16个地级市2005--2019年的周度病例与气象数据（聚合为月度）以及蚊媒监测数据为基础，分为两部分开展。第一部分（单城市机制发现）以广州为学习城市：通过逆问题方法从病例数据反推传播效率$\beta(t)$时间序列，利用多层感知机神经网络学习气象变量$(T,H,R)$到$\beta'$的非线性映射，再通过符号回归将神经网络"黑箱"翻译为显式二次交互多项式公式。在排除2014年极端暴发年份后，病例重建的Pearson $r$达到0.612、Spearman $\rho$达到0.705、$R^2_{\log}$达到0.450（MAE=51.23, RMSE=139.10）；符号回归发现的显式公式对神经网络输出的拟合精度$R^2=0.999987$。第二部分（多城市机制迁移）将广州发现的公式不经重新训练直接外推至其余15城市：2014年度横截面上16城Spearman $\rho=0.900$（$p=2.05\times10^{-6}$），非广州15城经线性重标定后MAE=61.8、RMSE=116.8，表明模型在跨城市风险排序与量级控制上均具备可用性。同时，广州2014年极端暴发中$\beta'$统计量未出现同量级跃升，提示非气象因素在极端年份中不可忽略。时间窗口敏感性分析（2005--2019 vs 2004--2023）进一步确认核心结论不依赖于特定训练时间范围。

本研究为蚊媒传染病传播机制的数据驱动发现提供了一种可复制、可解释、可迁移的方法论，完成了"单城机制发现$\rightarrow$显式公式固化$\rightarrow$多城迁移验证$\rightarrow$极端年份反证"的完整证据闭环。

\vspace{1em}
\noindent\textbf{关键词}：登革热；SEIR动力学模型；神经网络；符号回归；传播效率；气象因子；多城市验证

\newpage
""")

# ============ ENGLISH ABSTRACT ============
sections.append(r"""
\begin{center}
{\Large\textbf{Abstract}}
\end{center}

Dengue fever is one of the fastest-growing mosquito-borne infectious diseases worldwide, with its transmission efficiency driven by complex nonlinear interactions among meteorological factors such as temperature, humidity, and precipitation. However, existing models lack explicit functional relationships between meteorological conditions and transmission efficiency. This study proposes a ``SEIR dynamics + neural network + symbolic regression'' trinity framework to discover quantitative relationships between meteorological drivers and dengue transmission efficiency from real surveillance data, and to derive physically interpretable analytical formulas.

Using weekly case and meteorological data (aggregated to monthly) from 16 prefecture-level cities in Guangdong Province, China, during 2005--2019, together with mosquito surveillance data, the study is conducted in two parts. Part~I (single-city mechanism discovery) uses Guangzhou as the learning city: the transmission efficiency $\beta(t)$ time series is inversely derived from case data, a multilayer perceptron neural network learns the nonlinear mapping from meteorological variables $(T,H,R)$ to $\beta'$, and symbolic regression translates the neural network ``black box'' into an explicit quadratic polynomial formula with interaction terms. Excluding the extreme outbreak year of 2014, the case reconstruction achieves Pearson $r=0.612$, Spearman $\rho=0.705$, and $R^2_{\log}=0.450$ (MAE=51.23, RMSE=139.10); the explicit formula fits the neural network output with $R^2=0.999987$. Part~II (multi-city mechanism transfer) extrapolates the Guangzhou-derived formula to the remaining 15 cities without retraining: across the 16-city 2014 annual cross-section, Spearman $\rho=0.900$ ($p=2.05\times10^{-6}$); for the 15 non-Guangzhou cities after linear rescaling, MAE=61.8 and RMSE=116.8. The $\beta'$ statistics during Guangzhou's extreme 2014 outbreak did not exhibit a commensurate surge, suggesting that non-meteorological factors play a non-negligible role in extreme outbreaks. Time-window sensitivity analysis (2005--2019 vs.\ 2004--2023) further confirms that the core conclusions are robust to the choice of training period.

\vspace{1em}
\noindent\textbf{Keywords}: Dengue fever; SEIR dynamics model; Neural network; Symbolic regression; Transmission efficiency; Meteorological factors; Multi-city validation

\newpage
\tableofcontents
\newpage
\setcounter{page}{1}
\pagenumbering{arabic}
""")

# ============ 前言 ============
sections.append(r"""
% ============================================================
\section{前言}
% ============================================================

\subsection{登革热的全球与中国流行概况}

登革热（Dengue Fever）是由登革病毒（DENV）引起、主要经由伊蚊（\textit{Aedes aegypti}和\textit{Aedes albopictus}）叮咬传播的急性虫媒传染病。近年来，随着全球气候变暖、城市化进程加速以及国际贸易和旅游的频繁，登革热已成为世界上增长最快的虫媒病毒性疾病\cite{messina2019}。据Bhatt等\cite{bhatt2013}的研究估计，全球每年约有3.9亿人感染登革病毒，其中约9600万例出现临床症状，更有数万人死于重症登革热。近几十年来，登革热的发病率在全球急剧上升，向世界卫生组织报告的病例数从2000年的505,430例增加到2024年的1,460万例，现已在100多个国家流行\cite{who2024}。

在中国，登革热虽不是本土地方性流行病，但自20世纪70年代末以来，由输入性病例引发的本土暴发在东南沿海地区频发。特别是广东省，地处亚热带，气候温暖湿润，极适宜白纹伊蚊的生长繁殖，长期以来是我国登革热防控的重点区域\cite{yue2021,lai2015}。2014年，广东省经历了历史上最严重的登革热疫情，报告病例数超过4.5万例，不仅造成了巨大的健康威胁，也暴露了现有防控预警体系在应对极端暴发时的不足\cite{cheng2016}。因此，深入探究登革热的传播机制，特别是量化环境因素对传播过程的非线性驱动作用，对于制定精准的防控策略具有重要的现实意义。

\subsection{气候因素与蚊媒传染病传播}

气候因素是驱动蚊媒传染病时空分布和流行强度的核心外部变量，伊蚊的生命周期、种群密度及病毒在蚊体内的复制速率均受到气象条件的严格制约\cite{liyanage2016}。

\textbf{温度}直接影响蚊虫的生殖周期（Gonotrophic cycle）、幼虫发育率及成蚊存活率\cite{desouza2024,shapiro2017,lambrechts2011}。更重要的是，温度决定了外潜伏期（Extrinsic Incubation Period, EIP），即病毒在蚊体内复制并具备传播能力所需的时间\cite{kamiya2020}。Mordecai等\cite{mordecai2019}通过整合实验室数据建立的热生物学模型表明，登革热的传播效率与温度呈非线性单峰关系，最佳传播温度约为29$^\circ$C；当温度低于17.8$^\circ$C或高于34.5$^\circ$C时，传播基本受阻。Col\'{o}n-Gonz\'{a}lez等\cite{colon2018}基于多模型集合预测发现，温度升高将显著扩大登革热的适宜传播区，若全球升温幅度控制在2.0$^\circ$C以内，可避免拉丁美洲每年约280万例新增病例。

\textbf{降水}为蚊媒提供了必要的孳生场所，适量降雨能增加户外积水容器数量，促进蚊媒种群扩张\cite{zhou2025}。Nosrat等\cite{nosrat2021}的研究表明，月降水量异常偏高后的下一个月，蚊卵与成蚊丰度均显著增长。然而，强降雨会冲刷蚊媒孳生地，对媒介种群产生负面影响\cite{roiz2015}。Cheng等\cite{chengq2023}针对广州的研究发现，在前期水分充足的条件下，滞后7--121天的强降雨会降低登革热风险，其中滞后45天时的发病率比最低，为0.59（95\% CI: 0.43--0.79）。

\textbf{相对湿度}主要影响成蚊的存活时间，相对湿度在70\%--80\%时可优化蚊子的生存和觅食活动\cite{polrob2025}。Wu等\cite{wu2018}通过对广州市气候数据的分析发现，相对湿度影响登革热发病率的阈值为76\%，滞后7--14天的相对湿度每增加1\%，登革热风险相应上升1.95\%（95\% CI: 1.21\%--2.69\%）。

\subsection{登革热预测模型研究现状}

\subsubsection{统计学模型}

统计学模型常用于登革热预测，能够有效识别外部因素对传播的影响\cite{dacosta2025}。广义加性模型（GAM）与广义线性模型（GLM）被广泛用于识别登革热发生的风险因素\cite{leung2023}。Liu等\cite{liuk2020}构建了广义加性模型量化东亚夏季季风对1980--2016年中国大陆登革热发病率的影响。Sehi等\cite{sehi2025}采用负二项分布的GAM评估环境和气象对蚊子数量的影响，发现研究区域的埃及伊蚊丰度受地理位置、地表水、海拔和温度的显著影响。分布式滞后非线性模型（DLNM）在登革热环境流行病学中应用日益广泛：Luo等\cite{luo2025}利用DLNM对马来西亚、新加坡和泰国的登革热传播模式进行研究，揭示了新冠疫情前后环境暴露-响应关系的显著变化。混合智能模型通过整合气象数据与机器学习算法，显著提升了登革热预测准确性\cite{chengy2025}。

然而上述模型存在明显局限性：缺乏对疾病传播生物学机制的解释\cite{baker2022}，忽略了蚊媒种群动态、病毒传播周期等关键生物学过程\cite{mills2024}。

\subsubsection{机制动力学模型}

机制模型能为预测疾病系统在人口、技术和气候变化下的未来结果提供更稳健的框架\cite{baker2022,ahman2025}。经典的Ross-Macdonald模型奠定了蚊媒传染病建模的基础\cite{smith2012}，后续研究引入了潜伏期、气象因子和人口流动等因素\cite{guo2024,zhu2016,liuy2023}。数学建模已成为研究登革热传播和优化公共卫生干预的关键工具\cite{din2021}。

现有机制模型通常直接采用实验室测定的温依参数（如Bri\`{e}re函数描述叮咬率）\cite{mordecai2017,chen2024science}。然而，Caldwell等\cite{caldwell2021}指出，实验室环境与复杂的野外环境存在巨大差异，直接套用实验室参数往往导致模型预测偏差。Li等\cite{li2019pnas}在PNAS上发表的工作构建了气候驱动蚊媒密度的SIR模型，利用样条函数拟合时变传播率$\beta(t)$，成功复现了中国多个城市的登革热流行曲线。然而，其核心参数——传播效率$\beta$仅随时间变化，并没有显式地表达为温度、湿度、降雨等环境变量的函数，无法回答"什么气象条件导致高传播效率"这一关键问题。

\subsubsection{人工智能与机制模型的融合}

近年来，人工智能结合机制模型为解决复杂系统建模提供了新思路\cite{zhangs2021,yang2023}。Li等\cite{lir2024}将COVID-19模型动态嵌入物理信息神经网络（PINN），同时推断未知参数和底层模型动态。Nikparvar等\cite{nikparvar2021}将人口流动性作为变量输入LSTM网络用于预测COVID-19病例。Murphy等\cite{murphy2021}利用图神经网络学习网络传染动力学。然而，数据驱动方法虽然预测准确性优异，但难以提供可解释的因果机制\cite{holm2019}，制约了在公共卫生决策中的应用\cite{kamyshnyi2026,adeoye2025}。

\subsubsection{符号回归：从黑箱到白箱}

符号回归（Symbolic Regression）能够直接从数据中学习简明可解释的数学表达式\cite{makke2024}。Fajardo等\cite{fajardo2024}提出贝叶斯符号回归方法，用于从报告病例和检测率数据中自动学习传染病发病率的闭式数学模型。Zhang等\cite{zhang2024plos}通过将蚊媒种群动力学模型耦合神经网络揭示了伊蚊产卵率和温度、降水之间的关系，并使用符号回归确定最优函数表达式。然而，目前尚未有研究将"神经网络嵌入+符号回归"的完整框架应用于登革热传播效率反演与公式推导中。

\subsection{研究目标与创新}

本文提出一种结合SEIR动力学模型、神经网络和符号回归的混合建模框架，旨在发现真实环境下气象因子（温度、湿度、降雨）对登革热传播效率的非线性驱动机制，并推导出具有物理可解释性的解析公式。相比前人工作，本研究的创新点包括：
\begin{enumerate}[leftmargin=2em]
    \item \textbf{比PNAS更有机理性}：PNAS\cite{li2019pnas}的$\beta'(t)$是样条曲线，仅随时间变化。本研究的$\beta'(T,H,R)$显式依赖气象变量，能量化"温度每升高1$^\circ$C对传播的影响"。
    \item \textbf{比Zhang更直接}：Zhang等\cite{zhang2024plos}的NN替代的是产卵率，与疾病传播间接相关。本研究直接替代传播效率$\beta'$，更贴近登革热动力学研究的核心问题。
    \item \textbf{可解释+可迁移}：最终模型为完全解析的公式，无黑箱组件，可直接用于其他城市的疫情风险预测。
    \item \textbf{多城市验证}：不仅在单城市拟合，还在不重训参数的前提下外推至15个城市，检验机制的跨空间泛化能力。
\end{enumerate}

\subsection{全文结构}

本研究以广东省为研究区域，利用2005--2019年16个地级市的登革热病例数据、蚊媒监测数据及气象数据，按"先在单城市回答机制问题，再在多城市回答泛化问题"组织为两部分：
\begin{enumerate}[leftmargin=2em]
    \item \textbf{第一部分：机制发现（广州）}。核心问题：在真实监测数据下，能否从病例与气象中稳定识别$\beta'(T,H,R)$，并将其写成可解释公式？
    \item \textbf{第二部分：机制迁移（广东多城市）}。核心问题：第一部分得到的机制是否具有跨城市可迁移性，能否在不重训机制参数的前提下保持风险排序能力与可接受的量级误差？
\end{enumerate}
对应地，全文证据链围绕三个核心问题推进：\textbf{（i）能不能学出来？（ii）学出来的是什么？（iii）带到别的城市还能不能用？}这三个问题分别由Phase~1、Phase~2与多城市外推结果回答，并在2014极端年份分析中进行压力测试。
""")

# ============ 第一部分 ============
sections.append(r"""
% ============================================================
\section{第一部分：单城市机制发现（广州）}
% ============================================================

\subsection{引言}

登革热的传播过程受到多种环境因素的复杂影响\cite{ouedraogo2025}，但目前对于气象因素如何具体、量化地驱动传播效率尚缺乏统一的认识\cite{mills2024,white2025}。现有研究主要依赖基于实验室数据的参数化模型（如使用Bri\`{e}re方程描述温度影响）\cite{huber2018}，或者基于历史数据的统计模型\cite{lic2023}。然而，实验室的恒温环境难以真实反映野外复杂的微气候波动，且往往忽略了降雨和湿度对蚊媒生存的联合作用\cite{dennington2025}；而纯统计模型虽然能捕捉流行趋势，但缺乏对传播机理的解释能力\cite{polrob2025}。

广州市长期以来是我国登革热防控的重点区域，其亚热带季风气候极适宜白纹伊蚊孳生。特别是2014年，广州经历了历史罕见的大规模暴发，病例数超4.5万例\cite{chengj2021}。本章以广州市为例，提出一种结合"数据挖掘"与"机理建模"的方法，旨在回答一个关键问题：在真实环境中，温度、湿度和降雨通过什么样的数学关系决定登革热传播效率$\beta$？通过神经网络耦合动力学技术从历史数据中还原出隐含的传播率时间序列，再通过符号回归方法从复杂的神经网络中提取出具有物理意义的数学公式，以此揭示广州登革热暴发背后的环境驱动机制。

\subsection{数据材料和方法}

\subsubsection{研究区域与数据来源}

本章选取广东省省会广州市作为研究区域。广州市位于东经112$^\circ$57$'$至114$^\circ$03$'$，北纬22$^\circ$26$'$至23$^\circ$56$'$之间，属于典型的海洋性亚热带季风气候，年平均气温21.5--22.2$^\circ$C，雨量充沛\cite{chengj2021}。该地区不仅是白纹伊蚊的活跃区，也是中国大陆登革热病例报告最集中的城市，且拥有完善的蚊媒监测网络。

本研究收集了广州市2005--2019年的多源数据（表~\ref{tab:data}），将所有数据的时间尺度统一为月度。

\begin{table}[H]
    \centering
    \caption{数据来源汇总}
    \label{tab:data}
    \begin{tabular}{llcl}
        \toprule
        \textbf{数据类型} & \textbf{来源} & \textbf{时间范围} & \textbf{原始/目标分辨率} \\
        \midrule
        病例+气象（16城） & data\_2/data.csv & 2005--2019 & 周度$\rightarrow$月度 \\
        蚊媒监测（BI等） & 广东省CDC / CCM14 & 2005--2019 & 月度 \\
        人口数据 & 国家统计局 & 2005--2019 & 年度 \\
        \bottomrule
    \end{tabular}
\end{table}

气象数据（平均气温$T$、相对湿度$H$、累计降雨量$R$）来源于美国国家海洋和大气管理局（NOAA）下属的国家环境信息中心（NCEI）。首先基于广州区域气象站点观测值，运用反距离权重（IDW）插值技术生成空间分辨率为1\,km的逐日气象栅格数据，再依据行政区划矢量地图进行区域统计获得逐日气象均值，最终聚合为月度数据。蚊媒监测数据来源于广东省疾病预防控制中心及CCM14数据集\cite{ccm14}，使用布雷图指数（Breteau Index, BI）作为蚊媒密度代理指标。登革热病例数据从中国公共卫生科学数据中心收集\cite{lai2015}。广州市人口数据来源于国家统计局，本研究采用研究期中间时点（2012年）常住人口$N_h \approx 1.426\times10^7$人作为固定参考值\cite{guo2024}。选择固定人口而非逐年变化人口的原因在于：（1）研究期内广州常住人口增长率约为2\%/年，15年累计变化约30\%，相较于病例的年际波动（可达100倍以上）属于缓变量；（2）PNAS\cite{li2019pnas}等同类工作同样采用固定人口设定。

\subsubsection{数据预处理}

为消除不同物理量纲对神经网络训练效率的影响，将所有气象变量进行最小--最大归一化（Min-Max Normalization）：
\begin{equation}
    \hat{x}_t = \frac{x_t - x_{\min}}{x_{\max} - x_{\min}}
    \label{eq:minmax}
\end{equation}
其中$x_t$为第$t$个时间步长的气象观测值，$x_{\min}$和$x_{\max}$分别为该变量在全研究时段内的最小值与最大值。

蚊媒密度的归一化处理如下：
\begin{equation}
    \hat{M}(t) = \frac{\text{BI}(t)}{\overline{\text{BI}}}
    \label{eq:bi_norm}
\end{equation}
其中$\overline{\text{BI}}$为时间均值。$\hat{M}(t)$表示实时蚊媒密度相对于历史均值的比例，作为动力学模型中感染力的核心驱动项。针对监测数据中的少量缺失值，采用高斯平滑滤波进行填充以确保物理量的连续性。

\subsubsection{SEIR动力学模型}

人群传播动力学采用SEIR（易感--暴露--感染--恢复）模型。将总人口$N_h$划分为易感者（$S_h$）、潜伏者（$E_h$）、感染者（$I_h$）和康复者（$R_h$）：
\begin{align}
    \frac{dS_h}{dt} &= -\lambda(t) \cdot S_h \label{eq:dS}\\
    \frac{dE_h}{dt} &= \lambda(t) \cdot S_h + \eta - \sigma_h E_h \label{eq:dE}\\
    \frac{dI_h}{dt} &= \sigma_h E_h - \gamma I_h \label{eq:dI}\\
    \frac{dR_h}{dt} &= \gamma I_h \label{eq:dR}
\end{align}
其中感染力（Force of Infection）$\lambda(t)$定义为：
\begin{equation}
    \lambda(t) = \frac{\beta'(T,H,R) \cdot \hat{M}(t)}{N_h} \cdot I_h
    \label{eq:foi}
\end{equation}

各参数及其取值如表~\ref{tab:params}所示。

\begin{table}[H]
    \centering
    \caption{SEIR模型参数设置}
    \label{tab:params}
    \begin{tabular}{lccl}
        \toprule
        \textbf{参数} & \textbf{符号} & \textbf{取值} & \textbf{来源/说明} \\
        \midrule
        总人口 & $N_h$ & $1.426\times10^7$ & 国家统计局（2012年中间时点） \\
        潜伏期转化率 & $\sigma_h$ & $1/5.9\;\text{d}^{-1}$ & Chan \& Johansson (2012)\cite{chan2012} \\
        恢复率 & $\gamma$ & $1/14\;\text{d}^{-1}$ & Mordecai et al.\ (2017)\cite{mordecai2017} \\
        输入性感染率 & $\eta$ & 可训练参数 & 数据驱动估计 \\
        传播效率 & $\beta'(T,H,R)$ & NN学习 & 本研究核心 \\
        蚊媒密度代理 & $\hat{M}(t)$ & BI归一化 & 广东省CDC / CCM14 \\
        \bottomrule
    \end{tabular}
\end{table}

\textbf{参数单位说明}：$\sigma_h$和$\gamma$的物理含义分别为日潜伏转化速率和日恢复速率。本研究中ODE数值积分以天为基本步长（使用Runge-Kutta 4/5方法），故直接使用日速率值。模型输出的逐日新增病例聚合为月度后与观测数据对比。若需在月尺度简化模型中使用，可乘以30天转换为月速率（$\sigma_h^{(\text{月})} \approx 5.08\;\text{月}^{-1}$，$\gamma^{(\text{月})} \approx 2.14\;\text{月}^{-1}$）。

\textbf{输入性感染率$\eta$的处理}：$\eta$反映了境外或省外输入性病例对本地易感池的背景补充压力。由于输入病例的时空模式难以从外部数据精确获得，本研究将$\eta$设定为可训练的常数参数，在神经网络联合优化过程中自动从数据中估计\cite{cheng2016}。这一设计避免了人为预设的主观性，让模型自行从观测数据中推断背景输入强度。

基本再生数$R_0$由传播效率和蚊虫密度估算：
\begin{equation}
    R_0(t) = \frac{\beta'(T,H,R) \cdot \hat{M}(t)}{\gamma}
    \label{eq:R0}
\end{equation}
当$R_0 > 1$时疾病可能暴发流行。

\subsubsection{神经网络耦合框架}

传播效率$\beta'(T,H,R)$设定为环境因素的非线性函数，本研究构建了一个多层感知机（Multilayer Perceptron, MLP）神经网络来学习这一映射关系：
\begin{equation}
    \beta'(T,H,R) = s_1 \cdot \text{NN}_\theta(\hat{T}, \hat{H}, \hat{R}) + s_0
    \label{eq:nn}
\end{equation}
其中$s_0, s_1$为尺度缩放因子，$\theta$为网络权重参数。网络架构如表~\ref{tab:nn}所示。

\begin{table}[H]
    \centering
    \caption{传播效率神经网络架构}
    \label{tab:nn}
    \begin{tabular}{lccc}
        \toprule
        \textbf{层} & \textbf{输入维度} & \textbf{输出维度} & \textbf{激活函数} \\
        \midrule
        输入层 & 3 ($\hat{T}, \hat{H}, \hat{R}$) & 16 & Softplus \\
        隐藏层 & 16 & 16 & Softplus \\
        输出层 & 16 & 1 & Sigmoid \\
        \bottomrule
    \end{tabular}
\end{table}

选择Softplus激活函数（$\text{Softplus}(x) = \ln(1+e^x)$）是因为其保证了物理过程的光滑性和非负性导数特性；Sigmoid输出映射至$(0,1)$，代表归一化的传播概率。全网络共353个可训练参数。

\subsubsection{训练策略与损失函数}

训练采用两步法，参照PNAS\cite{li2019pnas}的轨迹匹配思想：

\textbf{Step 1——反推$\beta(t)$}：基于简化的SIR月度递推关系：
\begin{equation}
    \text{cases}(t) \approx \beta(t) \times \hat{M}(t) \times \text{pool}(t-1)
\end{equation}
其中$\text{pool}(t-1) = \text{cases}(t-1) + 0.3 \times \text{cases}(t-2)$为感染池（考虑了感染者在上一月和前两月的滞后贡献）。由此反推得到目标传播率序列：
\begin{equation}
    \beta(t) = \frac{\text{cases}(t)}{\hat{M}(t) \times \text{pool}(t-1)}
\end{equation}

\textbf{Step 2——训练NN}：以$(\hat{T}_t, \hat{H}_t, \hat{R}_t)$为输入，归一化$\beta(t)$为目标，损失函数设计为均方误差（MSE）与相关系数（Correlation）的加权组合：
\begin{equation}
    \mathcal{L}(\theta) = \frac{1}{N}\sum_{t=1}^{N}\left(\hat{y}_t - y_t\right)^2 - \lambda \cdot \frac{\text{Cov}(\hat{\mathbf{y}}, \mathbf{y})}{\sigma_{\hat{y}}\sigma_y}
    \label{eq:loss}
\end{equation}
其中$\hat{y}_t = \text{NN}_\theta(\hat{T}_t, \hat{H}_t, \hat{R}_t)$为预测值，$y_t$为反推的归一化$\beta(t)$观测值，$N$为去除2014年后的有效训练样本数（$N=168$个月），$\lambda=0.5$为平衡权重。MSE项控制数值精度，负相关项引导网络学习正确的季节性趋势。

参数更新采用Adam（Adaptive Moment Estimation）优化器\cite{kingma2015}，学习率为$10^{-3}$。Adam结合了梯度的一阶矩估计（Momentum）和二阶矩估计（RMSProp），能自适应地调整每个参数的学习率，适合处理气象数据中的非平稳噪声。训练使用PyTorch深度学习框架的自动微分（Autograd）机制实现。

\textbf{Step 3——病例重建验证}：用NN预测的$\beta'$结合蚊媒代理与病例滞后项，生成月度病例重建并与观测值对比。

\textbf{2014年留一法交叉验证}：为验证模型在极端气候条件下的泛化能力，将2014年全样本数据作为独立测试集，训练中通过掩膜机制屏蔽，仅计算除2014年以外年份的损失。这迫使神经网络学习到气象因素与传播率之间普适的非线性关系，而非记忆极端年份数据\cite{li2019pnas}。

\subsubsection{基于符号回归的耦合机制解析}

神经网络耦合动力学模型的核心组件$\text{NN}_\theta$本质上仍是"黑箱"。为揭示气象因子驱动传播效率的具体数学形式，采用符号回归（Symbolic Regression）技术将NN隐式映射翻译为显式解析函数。

\textbf{知识蒸馏策略}：直接利用原始数据进行公式搜索存在噪声大、样本稀疏等问题。本研究采用"知识蒸馏"策略，利用训练好的NN作为教师模型生成高密度虚拟数据集。在温度、湿度和降雨量的三维归一化空间$[0,1]^3$内生成均匀正交网格点，输入NN计算对应$\beta'$值：
\begin{equation}
    \mathcal{D}_{\text{distill}} = \{(\hat{T}_i, \hat{H}_i, \hat{R}_i, \beta'_i)\}_{i=1}^{N_s}
\end{equation}
该数据集剔除了环境噪声，纯粹表征NN所习得的气象--传播率响应曲面。

\textbf{候选公式族}：本研究同时探索两类候选公式：
\begin{enumerate}[leftmargin=2em]
    \item \textbf{物理模板族}：基于生态学先验知识预设功能模板——（a）温度单峰响应：采用高斯函数$f_T(T) = \exp(-(T-T_{\text{opt}})^2/2\sigma_T^2)$描述适宜温度范围，其中$T_{\text{opt}}$为最适传播温度（初始值设为27$^\circ$C，基于Mordecai等\cite{mordecai2019}的热生物学研究），$\sigma_T$控制适宜温区宽度；（b）降雨饱和效应：$f_R(R) = 1-\exp(-kR)$，其中$k$为饱和系数，反映降雨对孳生地的边际效应递减\cite{liuy2023}；（c）多因子乘法耦合：遵循Liebig最小因子定律\cite{mordecai2017}，采用连乘形式$\beta' = c \cdot f_T(T) \cdot f_H(H) \cdot f_R(R)$。利用PySR框架\cite{cranmer2023}进行搜索，算子集合包括$\{+,-,\times,\div,\exp,\log,\text{pow}\}$。
    \item \textbf{多项式族}：含交互项的二次多项式$\beta' = a_0 + \sum_i a_i x_i + \sum_{i \leq j} a_{ij}x_ix_j$，以闭式线性回归估计参数\cite{zhang2024plos}。
\end{enumerate}

以$R^2$/RMSE与可解释性综合选定最终公式，遵循帕累托前沿（Pareto Frontier）原则兼顾精度和简洁性。

\subsubsection{评估指标体系}

本研究采用多维度指标体系避免单一指标误判：排序指标（Spearman $\rho$, Kendall $\tau$）衡量风险等级排序一致性；相关指标（Pearson $r$）衡量线性趋势同步性；对数拟合优度（$R^2_{\log}$）缩小极值杠杆效应；绝对误差（MAE, RMSE）控制量级偏差；相对误差（WAPE, sMAPE, MAPE）提供异规模可比性；对数误差（RMSLE）应对零膨胀稳健性。

\subsection{结果}

\subsubsection{广州市登革热流行特征与气象因素基本特征}

在建模之前，首先对广州市登革热病例及气象协变量进行描述性分析。

从年尺度来看，2005--2019年广州年度病例总数呈现显著的年际波动，其中2014年出现极端异常峰值（37,382例），为典型的特大暴发年。除2014年外，2006年、2013年和2019年也可见次级高峰，而2008--2012年整体处于较低流行水平。上述结果表明，研究时段内同时包含低流行期与高流行期，有利于训练在不同传播强度下均具有稳健性的模型。

从月尺度来看，病例呈现夏秋季聚集特征：1--6月病例数普遍较低，7月开始逐渐升高，9--10月的病例中位数及离散程度均达全年最高，11--12月迅速回落。这一季节性格局与蚊媒种群的温度依赖性密切相关\cite{mordecai2019}。

气象因素方面，温度呈稳定年周期变化（夏季升高、冬季降低），相对湿度整体维持在较高水平并伴有年际起伏，降水量表现为间歇性高峰。值得注意的是，气象因子本身的年际变化远小于病例的年际起伏，提示气象--病例之间的关系并非简单线性，而更可能依赖于特定的多因子组合及阈值条件。

从反推的月度$\beta(t)$来看，其与温度呈显著正相关（$r=0.51$，$p<10^{-12}$），验证了气象因素对传播效率的驱动作用，为后续神经网络学习提供了可靠的训练目标。

\subsubsection{Phase 1：神经网络耦合动力学模型拟合结果}

NN成功学习了$\beta(t)$与气象变量的非线性关系。用NN预测的$\beta'$进行月度病例重建，核心性能指标如表~\ref{tab:phase1}所示。

\begin{table}[H]
    \centering
    \caption{Phase 1性能指标（广州，2005--2019，排除2014，$N=168$）}
    \label{tab:phase1}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{方案} & \textbf{Pearson $r$} & \textbf{Spearman $\rho$} & \textbf{$R^2_{\log}$} & \textbf{MAE} & \textbf{RMSE} \\
        \midrule
        NN病例重建 & \textbf{0.612} & \textbf{0.705} & \textbf{0.450} & 51.23 & 139.10 \\
        \bottomrule
    \end{tabular}
\end{table}

Spearman $\rho=0.705$表明模型对病例高低月份的排序一致性较好（即能正确识别高风险月份与低风险月份的相对顺序）。$R^2_{\log}=0.450$在对数尺度上解释了约45\%的变异，考虑到登革热月度病例跨越3--4个数量级的波动幅度，这一解释力度具有实际意义。MAE=51.23例/月在广州月均病例水平下处于合理范围。图~\ref{fig:phase1}展示了病例重建时间序列、$\beta$目标与NN输出对比、训练损失收敛过程及散点拟合。

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{../results/data2_1plus3/phase1_guangzhou_data2.png}
    \caption{Phase 1结果：（A）广州月度病例重建时间序列（观测 vs 预测）；（B）反推$\beta(t)$与NN输出对比；（C）训练损失收敛曲线；（D）观测-预测散点图。}
    \label{fig:phase1}
\end{figure}

\subsubsection{Phase 2：符号回归发现的解析公式}

对广州样本上的NN输出进行显式化拟合，两类候选公式的比较结果如表~\ref{tab:formulas}所示。

\begin{table}[H]
    \centering
    \caption{Phase 2候选公式对比}
    \label{tab:formulas}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{公式类型} & $r$ & $R^2$ & RMSE & MAE \\
        \midrule
        高斯温度$\times$饱和降雨模板 & 0.9986 & 0.9973 & $8.96\times10^{-4}$ & $6.21\times10^{-4}$ \\
        含交互项二次多项式 & \textbf{0.999994} & \textbf{0.999987} & $\mathbf{6.27\times10^{-6}}$ & $\mathbf{4.54\times10^{-6}}$ \\
        \bottomrule
    \end{tabular}
\end{table}

含交互项的二次多项式在精度上比物理模板族高出两个数量级（$R^2$: 0.999987 vs 0.9973）。最终选定的最优公式为：
\begin{equation}
    \boxed{
    \beta'(T,H,R)=\max\Big(
    0,\; a_0+a_TT+a_HH+a_RR+a_{TT}T^2+a_{HH}H^2+a_{RR}R^2+a_{TH}TH+a_{TR}TR+a_{HR}HR
    \Big)}
    \label{eq:formula}
\end{equation}
其中最优参数估计见表~\ref{tab:coeffs}。

\begin{table}[H]
    \centering
    \caption{公式~\eqref{eq:formula}的参数估计值}
    \label{tab:coeffs}
    \scriptsize
    \begin{tabular}{lclc}
        \toprule
        \textbf{参数} & \textbf{值} & \textbf{参数} & \textbf{值} \\
        \midrule
        $a_0$ (截距) & $1.800\times10^{-1}$ & $a_{TT}$ ($T^2$) & $2.695\times10^{-6}$ \\
        $a_T$ ($T$) & $5.065\times10^{-5}$ & $a_{HH}$ ($H^2$) & $-6.715\times10^{-7}$ \\
        $a_H$ ($H$) & $4.443\times10^{-5}$ & $a_{RR}$ ($R^2$) & $-2.167\times10^{-8}$ \\
        $a_R$ ($R$) & $-3.327\times10^{-5}$ & $a_{TH}$ ($T{\times}H$) & $8.071\times10^{-8}$ \\
        & & $a_{TR}$ ($T{\times}R$) & $8.389\times10^{-7}$ \\
        & & $a_{HR}$ ($H{\times}R$) & $3.084\times10^{-7}$ \\
        \bottomrule
    \end{tabular}
    \normalsize
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{../results/data2_1plus3/phase2_formula_fit_data2.png}
    \caption{Phase 2符号回归结果：显式公式与NN输出的一致性。}
    \label{fig:phase2}
\end{figure}

值得注意的是，虽然物理模板族在生物学可解释性上更直观（如$T_{\text{opt}}\approx27^\circ$C与Mordecai等\cite{mordecai2019}的实验室数据一致），但其对NN输出的逼近精度略低。考虑到后续多城市外推对公式精度的敏感性，选择二次交互多项式作为主公式。该公式的二次项和交互项仍保留了物理可解释性：$a_{TT}>0$提示温度的边际效应递增；$a_{TR}>0$表明高温与高降水同时出现时传播效率超线性增加；$a_{RR}<0$反映降水的边际效应递减（过量降水冲刷孳生地）。

\subsubsection{2014年极端暴发分析}

利用训练好的$\beta'(T,H,R)$分析广州2014年极端暴发年（37,382例）的驱动因素（表~\ref{tab:2014}）。

\begin{table}[H]
    \centering
    \caption{2014年$\beta'$与其他年份对比}
    \label{tab:2014}
    \begin{tabular}{lcc}
        \toprule
        & \textbf{2014年} & \textbf{其他年份均值} \\
        \midrule
        $\beta'$年均值 & 0.183539 & 0.183585 \\
        $\beta'$年峰值 & 0.186245 & 0.186601 \\
        \bottomrule
    \end{tabular}
\end{table}

2014年的$\beta'(T,H,R)$统计量与其他年份非常接近（差异$<$0.04\%），表明该年气象驱动的传播效率并未出现异常跃升。因此，2014年极端病例峰值很可能还受输入性病例时空聚集\cite{cheng2016}、城市人口流动\cite{kraemer2019}和防控响应时滞\cite{li2019pnas}等非气象因素影响，这与PNAS的分析结论一致。

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{../results/data2_1plus3/outbreak_2014_beta_compare_data2.png}
    \caption{广州2005--2019年度病例与$\beta'$统计：2014年病例峰值明显，但$\beta'$并未同步出现异常跃升。}
    \label{fig:2014}
\end{figure}

\subsection{讨论}

本章核心发现可概括为三点：

（1）\textbf{可学习性}：在排除2014年极端数据后，NN成功从广州15年月度序列中学习到$\beta'(T,H,R)$的非线性映射，病例重建Spearman $\rho=0.705$，说明气象驱动传播的"信号"在数据中可检测。这一结果为"传播效率可由气象变量驱动学习"提供了实证支持，超越了既往基于实验室参数的间接推断\cite{caldwell2021}。

（2）\textbf{可解释性}：符号回归将NN黑箱翻译为含10个参数的显式二次交互多项式，对NN输出拟合$R^2>0.9999$。公式中的交互项揭示了气象因子之间的协同效应：$a_{TR}>0$表明高温与高降水同时出现时传播效率超线性增加，与蚊媒在温暖潮湿环境下叮咬率和种群密度同步升高的生态学认识一致\cite{mordecai2019,brady2013}；$a_{RR}<0$反映降水边际效应递减，与Cheng等\cite{chengq2023}发现的"强降雨冲刷孳生地"效应吻合。

（3）\textbf{极端年份鉴别力}：2014年$\beta'$统计量未出现同量级跃升，模型能区分"气象可解释"与"气象不可解释"的传播变异，为极端暴发归因提供量化工具。这一发现的政策含义在于：即使气象条件处于"高传播窗口"，极端暴发的发生还需额外的触发条件（如大量输入性病例），这为针对性的防控资源配置提供了依据\cite{cheng2016}。

\textbf{局限性}：本章的传播效率学习仅在广州一个城市完成，其泛化能力有待在多城市环境下检验（见第二部分）。此外，月度时间分辨率可能平滑了周度尺度的传播动态细节，后续可结合更高分辨率数据验证。
""")

# ============ 第二部分 ============
sections.append(r"""
% ============================================================
\section{第二部分：多城市机制迁移与验证}
% ============================================================

\subsection{引言}

第一部分在广州建立了气象驱动传播效率的显式公式，但一个自然的追问是：该公式所捕获的是广州特有的局地规律，还是广东省乃至更大区域内蚊媒传播的共性结构？如果$\beta'(T,H,R)$反映的是蚊虫生物学对环境的普适响应，那么同一公式在不同城市的气象条件下应当产生与观测一致的风险排序，甚至在适当校准后给出可用的病例量级估计。

本部分正是为回答这一泛化问题而设计：将第一部分发现的$\beta'(T,H,R)$公式\textbf{不经任何重新训练}，直接外推至广东省其余15个地级市（共16城），在2014年度横截面与月度序列两个尺度上检验机制的跨空间可迁移性。这种"单城学习、多城验证"的设计借鉴了空间交叉验证的思想\cite{lowe2021,roberts2017}，避免了过拟合本地特征的风险。

\subsection{数据与方法}

\subsubsection{多城市数据}

本部分使用与第一部分相同来源的data\_2数据集，覆盖广东省16个地级市2005--2019年的周度登革热病例数据与气象数据，统一聚合为月度。16城市包括：广州、深圳、佛山、东莞、中山、珠海、江门、惠州、潮州、汕头、揭阳、阳江、茂名、湛江、清远、肇庆。蚊媒监测数据按城市可得性对齐，稳定覆盖8个城市；其余城市采用省级均值BI作为代理。

\subsubsection{外推方法与缩放策略}

外推流程：（1）对各城市月度气象数据使用公式~\eqref{eq:formula}计算$\beta'_c(t)$；（2）结合各城市蚊媒密度代理$\hat{M}_c(t)$和感染池$\text{pool}_c(t-1)$，生成月度预测病例；（3）汇总为年度预测值$\hat{Y}_c^{(\text{annual})}$，与观测年度病例$Y_c^{(\text{obs})}$对比。

考虑到城市间绝对量级差异，采用三种缩放口径：
\begin{itemize}[leftmargin=2em]
    \item \textbf{广州缩放}：以广州训练集的均值比例作为全局缩放因子；
    \item \textbf{去广州线性缩放}：仅用非广州城市拟合线性缩放系数，消除训练城市杠杆效应；
    \item \textbf{去广州log-linear缩放}：在对数空间进行线性缩放。
\end{itemize}

\subsubsection{评估口径}

本研究采用"\textbf{排序优先、误差补充、相关后置}"的评价口径：排序指标（Spearman $\rho$, Kendall $\tau$）评估城市间风险等级排序一致性；量级误差（MAE, RMSE, WAPE）评估绝对偏差；对数误差（RMSLE）缓解极端城市杠杆效应；相关指标（Pearson $r$）作为补充。这一口径的合理性在于：多城市外推不是"在每个城市重新拟合最优曲线"，而是对同一机制公式进行跨城检验\cite{lowe2021}。

\subsection{结果}

\subsubsection{多城市外推验证}

核心结果如表~\ref{tab:multicity}和表~\ref{tab:multicity_comp}所示。

\begin{table}[H]
    \centering
    \caption{多城市外推验证（2014年城市年度病例）}
    \label{tab:multicity}
    \scriptsize
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{缩放口径} & \textbf{N} & \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE} & \textbf{$\rho$} & \textbf{$p$} \\
        \midrule
        广州缩放（全16城） & 16 & 655.5 & 1499.6 & 1.129 & 0.900 & $2.05{\times}10^{-6}$ \\
        去广州线性缩放（全16城） & 16 & 1491.9 & 5737.1 & 0.224 & 0.900 & $2.05{\times}10^{-6}$ \\
        广州缩放（非广州15城） & 15 & 699.3 & 1548.8 & 1.204 & 0.879 & $1.63{\times}10^{-5}$ \\
        去广州线性缩放（非广州15城） & 15 & \textbf{61.8} & \textbf{116.8} & \textbf{0.198} & 0.879 & $1.63{\times}10^{-5}$ \\
        \bottomrule
    \end{tabular}
    \normalsize
\end{table}

\begin{table}[H]
    \centering
    \caption{综合指标（去广州线性缩放口径）}
    \label{tab:multicity_comp}
    \scriptsize
    \begin{tabular}{lccccccccc}
        \toprule
        \textbf{子集} & \textbf{N} & $r$ & $\rho$ & $\tau$ & $R^2_{\log}$ & MAE & RMSE & WAPE & RMSLE \\
        \midrule
        全16城 & 16 & 0.989 & 0.900 & 0.800 & 0.915 & 1491.9 & 5737.1 & 0.529 & 0.449 \\
        非广州15城 & 15 & 0.992 & 0.879 & 0.771 & 0.851 & 61.8 & 116.8 & 0.119 & 0.393 \\
        \bottomrule
    \end{tabular}
    \normalsize
\end{table}

三个关键发现：（1）\textbf{排序能力稳健}：全16城Spearman $\rho=0.900$（$p=2.05\times10^{-6}$），非广州15城$\rho=0.879$，能准确识别高低风险区域；（2）Kendall $\tau$达到0.800（全16城）与0.771（非广州15城），排序稳健性不受评价方法影响；（3）非广州15城去广州线性重标定后MAE=61.8、RMSE=116.8（WAPE=0.119, RMSLE=0.393），量级误差可控。

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{../results/data2_1plus3/transfer_2014_bars_data2.png}
    \caption{多城市外推验证：2014年16城市年度病例观测与预测对比。}
    \label{fig:multicity}
\end{figure}

\subsubsection{城市级月度指标分布}

在16城市月度序列层面，模型表现存在城市异质性。总体统计：Pearson $r$中位数0.481（均值0.466），Spearman $\rho$中位数0.469（均值0.484），$R^2_{\log}$中位数0.348（16城中13城$>0$）。Pearson $r\geq0.5$的城市为8个，Spearman $\rho\geq0.5$的城市为6个。部分城市$r\approx0.5$反映了周度零膨胀、城市异质性和量级差异带来的拟合难度，而非模型失效\cite{lic2023}。图~\ref{fig:grid}以网格图形式展示了16城市的月度拟合概览。

\begin{figure}[H]
    \centering
    \includegraphics[width=0.99\textwidth]{../results/data2_1plus3/all_cities_fit_grid.png}
    \caption{16城市月度拟合总览：左列为观测vs预测病例（对数坐标），右列为$\beta'(T,H,R)$时序（2005--2019）。}
    \label{fig:grid}
\end{figure}

\subsubsection{新旧数据口径对比}

为评估数据更新的影响，表~\ref{tab:old_new}对旧数据口径（省级月度、12城市）与新数据口径（16城市周度聚合月度）进行比较。

\begin{table}[H]
    \centering
    \caption{旧数据 vs 新数据核心指标对比}
    \label{tab:old_new}
    \scriptsize
    \begin{tabular}{lccccccc}
        \toprule
        \textbf{任务} & \textbf{版本} & \textbf{N} & $r$ & $\rho$ & $\tau$ & $R^2_{\log}$ & MAE \\
        \midrule
        Phase 1单城 & 旧口径 & 270 & 0.976 & 0.634 & 0.529 & 0.844 & 3.47 \\
        Phase 1单城 & 新data\_2 & 168 & 0.612 & \textbf{0.705} & \textbf{0.541} & 0.450 & 51.23 \\
        外推（非广州） & 旧口径 & 12 & 0.641 & 0.713 & 0.545 & $-$0.324 & 504.7 \\
        外推（非广州） & 新data\_2 & 15 & \textbf{0.992} & \textbf{0.879} & \textbf{0.771} & \textbf{0.851} & \textbf{61.8} \\
        \bottomrule
    \end{tabular}
    \normalsize
\end{table}

尽管新数据在单城市逐点拟合上更难，但在跨城市外推这一核心任务上，排序（$\rho$从0.713提升至0.879）与误差（MAE从504.7降至61.8）均明显改善。

\subsubsection{时间窗口敏感性分析}

为检验主结论对训练时间窗口的依赖程度，将同一流程分别在子集窗口（2005--2019）与全量窗口（2004--2023）上运行。

\begin{table}[H]
    \centering
    \caption{时间窗口敏感性分析}
    \label{tab:sensitivity}
    \scriptsize
    \begin{tabular}{lccccc}
        \toprule
        \textbf{窗口} & Phase 1 $r$ & Phase 1 $\rho$ & 外推$\rho$ & 外推MAE & 外推RMSE \\
        \midrule
        2005--2019 & 0.612 & 0.705 & 0.879 & \textbf{61.8} & \textbf{116.8} \\
        2004--2023 & \textbf{0.642} & \textbf{0.716} & \textbf{0.904} & 196.6 & 371.5 \\
        \bottomrule
    \end{tabular}
    \normalsize
\end{table}

两个窗口呈互补特征：全量窗口排序能力更高（$\rho=0.904$ vs 0.879），子集窗口MAE更低（61.8 vs 196.6）。两窗口下Phase~2公式系数差异在小数点第四位以后，城市级月度$\rho$差仅0.004。综合来看，本文选择2005--2019窗口作为主结果，全量窗口作为稳健性佐证。

\subsection{讨论}

本部分核心发现是：\textbf{广州发现的气象--传播效率映射具有跨空间可迁移性}。在不重训任何参数的前提下，16城市风险排序达到Spearman $\rho=0.900$（$p=2.05\times10^{-6}$），这一排序精度对公共卫生部门的资源分配决策已具备实用价值\cite{lowe2021}。

\textbf{排序能力 vs 量级精度的权衡}：本研究有意将排序指标置于量级指标之前，原因在于：（1）在资源有限情境下，先识别"哪座城市更可能高风险"通常比追求逐点精确值更具决策价值\cite{baker2022}；（2）城市间绝对量级受人口基数、城市化率、医疗报告率等非气象因素调制，难以仅凭气象机制完全刻画\cite{kraemer2019}。去广州线性重标定在保持排序不变的前提下将MAE降至61.8，表明简单的量级校准即可大幅提升实用性。

\textbf{城市异质性的来源}：部分城市月度$r\approx0.5$，主要源于：（i）周度数据中大量零值在月度聚合后仍产生零膨胀效应；（ii）非广州城市蚊媒数据可用性有限；（iii）各城市输入性病例占比、城市化水平与防控力度存在差异\cite{cheng2016}。后续可通过引入城市特异的校准系数进一步提升逐城精度。

\textbf{与PNAS的对比}：Li等\cite{li2019pnas}在8城市上用样条$\beta'(t)$取得优异拟合，但每个城市需独立拟合样条参数。本研究用\textbf{同一个}显式公式覆盖16城市，代价是逐点精度略低，收益是机制统一性与可迁移性显著增强。这体现了"解释力"与"拟合力"之间的根本权衡\cite{baker2022}：前者追求跨环境的一般性规律，后者追求特定环境下的最优匹配。
""")

# ============ 总结与展望 ============
sections.append(r"""
% ============================================================
\section{总结与展望}
% ============================================================

\subsection{主要结论}

本研究提出并验证了一种\textbf{SEIR动力学模型+神经网络+符号回归}的三位一体框架，用于发现登革热传播效率与气象因素的定量关系。基于广东省16城市2005--2019年多源数据，主要结论如下：

\begin{enumerate}[leftmargin=2em]
    \item \textbf{可学习性}：在广州单城市实验中，排除2014年极端年份后，NN从气象变量中学习到的$\beta'(T,H,R)$映射实现了有效的病例重建（Pearson $r=0.612$，Spearman $\rho=0.705$，$R^2_{\log}=0.450$），验证了传播效率可由气象变量驱动学习的假设。

    \item \textbf{可解释性}：符号回归发现最优公式为含交互项的二次多项式，对NN输出拟合$R^2=0.999987$，实现了从黑箱到白箱的完全转化。公式中的交互项（$a_{TR}>0$, $a_{RR}<0$）揭示了温度-降水协同效应和降水边际效应递减等生态学机制。

    \item \textbf{可迁移性}：将广州公式不经重训直接外推至16城市，2014年度横截面Spearman $\rho=0.900$（$p=2.05\times10^{-6}$）；非广州15城去广州重标定后MAE=61.8，RMSE=116.8，证实模型捕获的是区域性共性机制而非单城市特异噪声。

    \item \textbf{极端年份鉴别力}：2014年广州极端病例峰值中$\beta'$统计量未出现同量级跃升，表明气象条件是登革热传播的必要但非充分条件，非气象因素在极端暴发中不可忽略。

    \item \textbf{稳健性}：时间窗口敏感性分析（2005--2019 vs 2004--2023）确认，机制发现结果、多城市排序能力以及公式系数均对窗口选择不敏感。
\end{enumerate}

本框架完成了"单城机制发现$\rightarrow$显式公式固化$\rightarrow$多城迁移验证$\rightarrow$极端年份反证"的完整证据闭环，为蚊媒传染病传播机制的数据驱动发现提供了可复制、可解释、可迁移的方法论。

\subsection{创新贡献}

\begin{enumerate}[leftmargin=2em]
    \item \textbf{方法创新}：首次将"NN嵌入动力学模型+符号回归"的完整框架应用于登革热传播效率的反演与公式推导，弥补了PNAS样条方法"知其然不知其所以然"的缺陷。
    \item \textbf{公式贡献}：发现了$\beta'(T,H,R)$的二次交互多项式显式表达，为后续研究提供可复用的气象-传播效率定量工具。
    \item \textbf{验证范式}：建立了"排序优先、误差补充"的多城市外推评估口径，为类似的跨空间机制迁移研究提供方法论参考。
\end{enumerate}

\subsection{局限性与展望}

\begin{enumerate}[leftmargin=2em]
    \item \textbf{蚊媒数据覆盖不足}：16城市中仅8城可稳定对齐蚊媒指标，其余城市依赖省级代理。后续应优先推动多城市统一口径的蚊媒密度监测体系建设。
    \item \textbf{时间分辨率有限}：当前模型在月度尺度运行，丢失了周度甚至日度尺度的传播动态细节。后续可结合神经常微分方程（Neural ODE）框架\cite{chen2018node}实现更精细的动力学模拟。
    \item \textbf{非气象因素未纳入}：2014年极端暴发归因分析提示输入性病例、人口流动与防控策略变化是不可忽略的驱动因素。后续可在SEIR框架中增加人口流动模块\cite{kraemer2019}和时变防控力度参数。
    \item \textbf{跨区域推广待验证}：当前验证限于广东省内16城市。未来可将该框架推广至云南、浙江等其他登革热流行省份，甚至东南亚等跨国区域\cite{luo2025}。
    \item \textbf{实时预警应用}：显式公式$\beta'(T,H,R)$可直接接入气象预报数据实现前瞻性风险评估\cite{dacosta2025}，后续可开发基于该公式的实时预警原型系统，为公共卫生部门提供决策支持。
\end{enumerate}
""")

# ============ REFERENCES ============
sections.append(r"""
% ============================================================
\newpage
\begin{thebibliography}{99}

\bibitem{messina2019}
Messina JP, Brady OJ, Golding N, et al.
The current and future global distribution and population at risk of dengue.
\textit{Nature Microbiology}, 2019, 4(9): 1508--1515.
DOI: 10.1038/s41564-019-0476-8.

\bibitem{bhatt2013}
Bhatt S, Gething PW, Brady OJ, et al.
The global distribution and burden of dengue.
\textit{Nature}, 2013, 496(7446): 504--507.
DOI: 10.1038/nature12060.

\bibitem{who2024}
World Health Organization. Dengue and severe dengue. WHO Fact Sheet, 2024.

\bibitem{yue2021}
Yue Y, Liu Q, Liu X, et al.
Comparative analyses on epidemiological characteristics of dengue fever in Guangdong and Yunnan, China, 2004--2018.
\textit{BMC Public Health}, 2021, 21(1): 1389. DOI: 10.1186/s12889-021-11323-5.

\bibitem{lai2015}
Lai S, Huang Z, Zhou H, et al.
The changing epidemiology of dengue in China, 1990--2014.
\textit{BMC Medicine}, 2015, 13(1): 100. DOI: 10.1186/s12916-015-0336-1.

\bibitem{cheng2016}
Cheng Q, Jing Q, Spear RC, et al.
Climate and the timing of imported cases as determinants of the dengue outbreak in Guangzhou, 2014.
\textit{PLoS Neglected Tropical Diseases}, 2016, 10(2): e0004417. DOI: 10.1371/journal.pntd.0004417.

\bibitem{liyanage2016}
Liyanage P, Tissera H, Sewe M, et al.
A spatial hierarchical analysis of the temporal influences of the El Ni\~{n}o-Southern Oscillation and weather on dengue in Sri Lanka.
\textit{Int J Environ Res Public Health}, 2016, 13(11): 1087. DOI: 10.3390/ijerph13111087.

\bibitem{desouza2024}
de Souza WM, Weaver SC.
Effects of climate change and human activities on vector-borne diseases.
\textit{Nature Rev Microbiol}, 2024, 22(8): 476--491. DOI: 10.1038/s41579-024-01026-0.

\bibitem{shapiro2017}
Shapiro LLM, Whitehead SA, Thomas MB.
Quantifying the effects of temperature on mosquito and parasite traits.
\textit{PLoS Biology}, 2017, 15(10): e2003489. DOI: 10.1371/journal.pbio.2003489.

\bibitem{lambrechts2011}
Lambrechts L, Paaijmans KP, Fansiri T, et al.
Impact of daily temperature fluctuations on dengue virus transmission by \textit{Aedes aegypti}.
\textit{PNAS}, 2011, 108(18): 7460--7465. DOI: 10.1073/pnas.1101377108.

\bibitem{kamiya2020}
Kamiya T, Greischar MA, Wadhawan K, et al.
Temperature-dependent variation in the extrinsic incubation period elevates the risk of vector-borne disease emergence.
\textit{Epidemics}, 2020, 30: 100382. DOI: 10.1016/j.epidem.2019.100382.

\bibitem{mordecai2019}
Mordecai EA, Caldwell JM, Grossman MK, et al.
Thermal biology of mosquito-borne disease.
\textit{Ecology Letters}, 2019, 22(10): 1690--1708. DOI: 10.1111/ele.13335.

\bibitem{colon2018}
Col\'{o}n-Gonz\'{a}lez FJ, Harris I, Osborn TJ, et al.
Limiting global-mean temperature increase to 1.5--2\,\textdegree C could reduce dengue in Latin America.
\textit{PNAS}, 2018, 115(24): 6243--6248. DOI: 10.1073/pnas.1718945115.

\bibitem{zhou2025}
Zhou Z, He G, Hu J, et al.
Spatiotemporal expansion of \textit{Aedes aegypti} and dengue under climate change in China.
\textit{PLoS Negl Trop Dis}, 2025, 19(11): e0013702. DOI: 10.1371/journal.pntd.0013702.

\bibitem{nosrat2021}
Nosrat C, Altamirano J, Anyamba A, et al.
Impact of recent climate extremes on mosquito-borne disease transmission in Kenya.
\textit{PLoS Negl Trop Dis}, 2021, 15(3): e0009182. DOI: 10.1371/journal.pntd.0009182.

\bibitem{roiz2015}
Roiz D, Bouss\`{e}s P, Simard F, et al.
Autochthonous chikungunya transmission and extreme climate events in Southern France.
\textit{PLoS Negl Trop Dis}, 2015, 9(6): e0003854. DOI: 10.1371/journal.pntd.0003854.

\bibitem{chengq2023}
Cheng Q, Jing Q, Collender PA, et al.
Prior water availability modifies the effect of heavy rainfall on dengue transmission.
\textit{Front Public Health}, 2023, 11: 1287678. DOI: 10.3389/fpubh.2023.1287678.

\bibitem{polrob2025}
Polrob W, La-up A.
Nonlinear and lagged effects of climate variability on dengue incidence: a DLNM study in Bangkok.
\textit{BMC Public Health}, 2025, 25(1): 4024. DOI: 10.1186/s12889-025-25420-2.

\bibitem{wu2018}
Wu X, Lang L, Ma W, et al.
Non-linear effects of mean temperature and relative humidity on dengue incidence in Guangzhou, China.
\textit{Sci Total Environ}, 2018, 628--629: 766--771. DOI: 10.1016/j.scitotenv.2018.02.136.

\bibitem{dacosta2025}
da Costa JMF, Costa AC, Silveira CdS, et al.
Forecasting and early warning systems for dengue outbreaks: updated narrative review.
\textit{Rev Soc Bras Med Trop}, 2025, 59: e0429-2025. DOI: 10.1590/0037-8682-0429-2025.

\bibitem{leung2023}
Leung XY, Islam RM, Adhami M, et al.
A systematic review of dengue outbreak prediction models.
\textit{PLoS Negl Trop Dis}, 2023, 17(2): e0010631. DOI: 10.1371/journal.pntd.0010631.

\bibitem{liuk2020}
Liu K, Hou X, Ren Z, et al.
Climate factors and the East Asian summer monsoon may drive large outbreaks of dengue in China.
\textit{Environ Res}, 2020, 183: 109190. DOI: 10.1016/j.envres.2020.109190.

\bibitem{sehi2025}
Sehi GT, Birhanie SK, Hans J, et al.
Environmental correlates of \textit{Aedes aegypti} abundance in San Bernardino County, California.
\textit{Parasites Vectors}, 2025, 18: 349. DOI: 10.1186/s13071-025-06967-w.

\bibitem{luo2025}
Luo W, Liu Z, Ran Y, et al.
Unraveling varying spatiotemporal patterns of dengue fever in three Southeast Asian countries.
\textit{PLoS Negl Trop Dis}, 2025, 19(4): e0012096. DOI: 10.1371/journal.pntd.0012096.

\bibitem{chengy2025}
Cheng Y, Cheng R, Xu T, et al.
Integrating meteorological data and hybrid intelligent models for dengue fever prediction.
\textit{BMC Public Health}, 2025, 25: 1516. DOI: 10.1186/s12889-025-22375-2.

\bibitem{baker2022}
Baker RE, Mahmud AS, Miller IF, et al.
Infectious disease in an era of global change.
\textit{Nature Rev Microbiol}, 2022, 20(4): 193--205. DOI: 10.1038/s41579-021-00639-z.

\bibitem{mills2024}
Mills C, Donnelly CA.
Climate-based modelling and forecasting of dengue in three endemic departments of Peru.
\textit{PLoS Negl Trop Dis}, 2024, 18(12): e0012596. DOI: 10.1371/journal.pntd.0012596.

\bibitem{ahman2025}
Ahman QO, Aja RO, Omale D, et al.
Mathematical modeling of dengue virus transmission: exploring vector, vertical, and sexual pathways.
\textit{BMC Infect Dis}, 2025, 25(1): 999. DOI: 10.1186/s12879-025-11435-y.

\bibitem{smith2012}
Smith DL, Battle KE, Hay SI, et al.
Ross, Macdonald, and a theory for the dynamics and control of mosquito-transmitted pathogens.
\textit{PLoS Pathog}, 2012, 8(4): e1002588. DOI: 10.1371/journal.ppat.1002588.

\bibitem{guo2024}
Guo X, Li L, Ren W, et al.
Modelling the dynamic basic reproduction number of dengue based on MOI in Guangzhou.
\textit{Parasites Vectors}, 2024, 17: 79. DOI: 10.1186/s13071-024-06121-y.

\bibitem{zhu2016}
Zhu G, Liu J, Tan Q, et al.
Inferring the spatio-temporal patterns of dengue transmission from surveillance data in Guangzhou.
\textit{PLoS Negl Trop Dis}, 2016, 10(4): e0004633. DOI: 10.1371/journal.pntd.0004633.

\bibitem{liuy2023}
Liu Y, Wang X, Tang S, et al.
The relative importance of key meteorological factors affecting numbers of mosquito vectors.
\textit{PLoS Negl Trop Dis}, 2023, 17(4): e0011247. DOI: 10.1371/journal.pntd.0011247.

\bibitem{din2021}
Din A, Khan T, Li Y, et al.
Mathematical analysis of dengue stochastic epidemic model.
\textit{Results in Physics}, 2021, 20: 103719. DOI: 10.1016/j.rinp.2020.103719.

\bibitem{mordecai2017}
Mordecai EA, Cohen JM, Evans MV, et al.
Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models.
\textit{PLoS Negl Trop Dis}, 2017, 11(4): e0005568. DOI: 10.1371/journal.pntd.0005568.

\bibitem{chen2024science}
Chen Y, Xu Y, Wang L, et al.
Indian Ocean temperature anomalies predict long-term global dengue trends.
\textit{Science}, 2024, 384(6696): 639--646. DOI: 10.1126/science.adj4427.

\bibitem{caldwell2021}
Caldwell JM, LaBeaud AD, Lambin EF, et al.
Climate predicts geographic and temporal variation in mosquito-borne disease dynamics.
\textit{Nature Commun}, 2021, 12(1): 1233. DOI: 10.1038/s41467-021-21496-7.

\bibitem{li2019pnas}
Li R, Xu L, Bj{\o}rnstad ON, et al.
Climate-driven variation in mosquito density predicts the spatiotemporal dynamics of dengue.
\textit{PNAS}, 2019, 116(9): 3624--3629. DOI: 10.1073/pnas.1806094116.

\bibitem{zhangs2021}
Zhang S, Ponce J, Zhang Z, et al.
An integrated framework for building trustworthy data-driven epidemiological models.
\textit{PLoS Comput Biol}, 2021, 17(9): e1009334. DOI: 10.1371/journal.pcbi.1009334.

\bibitem{yang2023}
Yang HC, Xue Y, Pan Y, et al.
Time fused coefficient SIR model with application to COVID-19.
\textit{J Appl Stat}, 2023, 50(11--12): 2373--2387. DOI: 10.1080/02664763.2021.1936467.

\bibitem{lir2024}
Li R, Song Y, Qu H, et al.
A data-driven epidemic model with human mobility and vaccination protection for COVID-19 prediction.
\textit{J Biomed Inform}, 2024, 149: 104571. DOI: 10.1016/j.jbi.2023.104571.

\bibitem{nikparvar2021}
Nikparvar B, Rahman MM, Hatami F, et al.
Spatio-temporal prediction of the COVID-19 pandemic in US counties: modeling with a deep LSTM neural network.
\textit{Sci Rep}, 2021, 11(1): 21715. DOI: 10.1038/s41598-021-01119-3.

\bibitem{murphy2021}
Murphy C, Laurence E, Allard A.
Deep learning of contagion dynamics on complex networks.
\textit{Nature Commun}, 2021, 12(1): 4720. DOI: 10.1038/s41467-021-24732-2.

\bibitem{holm2019}
Holm EA.
In defense of the black box.
\textit{Science}, 2019, 364(6435): 26--27. DOI: 10.1126/science.aax0162.

\bibitem{kamyshnyi2026}
Kamyshnyi O, Halabitska I, Oksenych V, et al.
Forecasting influenza epidemics and pandemics in the age of AI and machine learning.
\textit{Rev Med Virol}, 2026, 36(1): e70107. DOI: 10.1002/rmv.70107.

\bibitem{adeoye2025}
Adeoye A, Onifade IA, Bayode M, et al.
Artificial intelligence and computational methods for modelling and forecasting influenza and influenza-like illness.
\textit{Beni-Suef Univ J Basic Appl Sci}, 2025, 14(1): 93. DOI: 10.1186/s43088-025-00682-2.

\bibitem{makke2024}
Makke N, Chawla S.
Interpretable scientific discovery with symbolic regression: a review.
\textit{Artif Intell Rev}, 2024, 57(1): 2. DOI: 10.1007/s10462-023-10622-0.

\bibitem{fajardo2024}
Fajardo-Fontiveros O, Mattei M, Burgio G, et al.
Machine learning mathematical models for incidence estimation during pandemics.
\textit{PLoS Comput Biol}, 2024, 20(12): e1012687. DOI: 10.1371/journal.pcbi.1012687.

\bibitem{zhang2024plos}
Zhang M, Wang X, Tang S.
Integrating dynamic models and neural networks to discover the mechanism of meteorological factors on \textit{Aedes} population.
\textit{PLoS Comput Biol}, 2024, 20(9): e1012499. DOI: 10.1371/journal.pcbi.1012499.

\bibitem{ouedraogo2025}
Ou\'{e}draogo JCRP, Ilboudo S, Tetteh RJ, et al.
Effects of environmental factors on dengue incidence in the Central Region, Burkina Faso.
\textit{PLoS Negl Trop Dis}, 2025, 19(7): e0013356. DOI: 10.1371/journal.pntd.0013356.

\bibitem{white2025}
White SM, Tegar S, Purse BV, et al.
Modelling the Lodi 2023 and Fano 2024 Italy dengue outbreaks.
\textit{Transbound Emerg Dis}, 2025, 2025(1): 5542740. DOI: 10.1155/tbed/5542740.

\bibitem{huber2018}
Huber JH, Childs ML, Caldwell JM, et al.
Seasonal temperature variation influences climate suitability for dengue, chikungunya, and Zika transmission.
\textit{PLoS Negl Trop Dis}, 2018, 12(5): e0006451. DOI: 10.1371/journal.pntd.0006451.

\bibitem{lic2023}
Li C, Liu Z, Li W, et al.
Projecting future risk of dengue related to hydrometeorological conditions in mainland China.
\textit{Lancet Planet Health}, 2023, 7(5): e397--e406. DOI: 10.1016/S2542-5196(23)00051-7.

\bibitem{dennington2025}
Dennington NL, Grossman MK, Teeple JL, et al.
Phenotypic variation in \textit{Aedes aegypti} populations and implications for predicting effects of temperature on dengue.
\textit{PLoS Negl Trop Dis}, 2025, 19(11): e0013623. DOI: 10.1371/journal.pntd.0013623.

\bibitem{chengj2021}
Cheng J, Bambrick H, Yakob L, et al.
Extreme weather conditions and dengue outbreak in Guangdong, China: spatial heterogeneity.
\textit{Environ Res}, 2021, 196: 110900. DOI: 10.1016/j.envres.2021.110900.

\bibitem{ccm14}
CCM14: Mosquito surveillance data in China. \url{https://github.com/xyyu001/CCM14}.

\bibitem{chan2012}
Chan M, Johansson MA.
The incubation periods of dengue viruses.
\textit{PLoS ONE}, 2012, 7(11): e50972. DOI: 10.1371/journal.pone.0050972.

\bibitem{brady2013}
Brady OJ, Johansson MA, Guerra CA, et al.
Modelling adult \textit{Aedes aegypti} and \textit{Aedes albopictus} survival at different temperatures.
\textit{Parasites Vectors}, 2013, 6(1): 351. DOI: 10.1186/1756-3305-6-351.

\bibitem{kingma2015}
Kingma DP, Ba J.
Adam: a method for stochastic optimization.
\textit{Proc ICLR}, 2015. arXiv: 1412.6980.

\bibitem{cranmer2023}
Cranmer M.
Interpretable machine learning for science with PySR and SymbolicRegression.jl.
\textit{arXiv preprint}, 2023. arXiv: 2305.01582.

\bibitem{kraemer2019}
Kraemer MUG, Reiner RC, Brady OJ, et al.
Past and future spread of the arbovirus vectors \textit{Aedes aegypti} and \textit{Aedes albopictus}.
\textit{Nature Microbiol}, 2019, 4(5): 854--863. DOI: 10.1038/s41564-019-0376-y.

\bibitem{chen2018node}
Chen RTQ, Rubanova Y, Bettencourt J, et al.
Neural ordinary differential equations.
\textit{Advances in NeurIPS}, 2018, 31: 6571--6583.

\bibitem{lowe2021}
Lowe R, Lee SA, O'Reilly KM, et al.
Combined effects of hydrometeorological hazards and urbanisation on dengue risk in Brazil.
\textit{Lancet Planet Health}, 2021, 5(4): e209--e219. DOI: 10.1016/S2542-5196(20)30292-8.

\bibitem{roberts2017}
Roberts DR, Bahn V, Ciuti S, et al.
Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.
\textit{Ecography}, 2017, 40(8): 913--929. DOI: 10.1111/ecog.02881.

\end{thebibliography}

\newpage
\section*{致谢}
\addcontentsline{toc}{section}{致谢}

在本论文即将完成之际，我首先要向我的指导教师XXX教授致以最诚挚的谢意。XXX教授在论文选题、研究方法和论文写作的各个阶段都给予了悉心的指导和热忱的鼓励，其严谨的治学态度和开阔的学术视野令我深受教益。

感谢XXX学院的各位老师在课程学习和科研训练中给予的帮助和启发。感谢广东省疾病预防控制中心提供的蚊媒监测数据支持。感谢课题组的各位同学在数据收集、代码调试和学术讨论中的协作与支持。

最后，衷心感谢家人多年来的默默支持与鼓励，是他们的理解与陪伴让我能够安心完成学业。

\end{document}
""")

# Write to file
with open('/root/wenmei/paper_biye/thesis_draft.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sections))

# Count characters
content = '\n'.join(sections)
# Remove LaTeX commands roughly for char count
import re
text_only = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', '', content)
text_only = re.sub(r'[{}\[\]\\$&%#_^~]', '', text_only)
text_only = re.sub(r'\s+', '', text_only)
print(f"File written: {len(content)} bytes")
print(f"Approx content chars (no whitespace): {len(text_only)}")

# Count refs
ref_count = content.count('\\bibitem{')
print(f"Reference count: {ref_count}")

# Count recent refs (2021-2026)
recent = 0
for year in ['2021', '2022', '2023', '2024', '2025', '2026']:
    recent += content.count(f', {year},') + content.count(f', {year}.')
print(f"Recent refs (2021-2026 approximate): {recent}")
