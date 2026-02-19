#!/usr/bin/env python3
"""Write all tex part files."""
import os
D = "/root/wenmei/paper_biye/tex_parts"
os.makedirs(D, exist_ok=True)

def w(name, content):
    with open(os.path.join(D, name), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {name}")

# Already written: 00_preamble.tex

w("01_titlepage.tex", r"""
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
""")

w("04_toc.tex", r"""
\newpage
\tableofcontents
\newpage
\setcounter{page}{1}
\pagenumbering{arabic}
""")

w("99_end.tex", r"\end{document}")

print("Simple parts done")
