# -*- coding: utf-8 -*-
with open('paper_draft/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace results intro + Phase 1 header
m1 = '\\section{结果}\n% ============'
m2 = '\\subsection{Phase 1: 传播效率学习}\n'
i1 = content.find(m1)
i2 = content.find(m2, i1)
assert i1 >= 0 and i2 >= 0, f"i1={i1}, i2={i2}"
i2e = i2 + len(m2)

nb = []
nb.append('\\section{结果}')
nb.append('% ============================================================')
nb.append('')
nb.append('本节按\u201c\\textbf{先机制、后公式、再迁移}\u201d的叙事顺序展开，')
nb.append('并在章节层面显式划分为两部分：')
nb.append('\\textbf{第一部分}（\\S\\ref{sec:part1}）在广州单城市回答\u201c能不能学出来\u201d与\u201c学出来的是什么\u201d；')
nb.append('\\textbf{第二部分}（\\S\\ref{sec:part2}）将机制固定后外推至多城市，回答\u201c带到别的城市还能不能用\u201d。')
nb.append('每一步结论都由上一环节的结果支撑，形成层层递进的证据链。')
nb.append('')
nb.append('% ------------------------------------------------------------')
nb.append('\\subsection{第一部分：单城市机制发现（广州）}')
nb.append('\\label{sec:part1}')
nb.append('% ------------------------------------------------------------')
nb.append('')
nb.append('本部分聚焦于广州2005--2019年数据，依次完成三个子任务：')
nb.append('（i）从病例反推传播效率$\\beta(t)$并验证其与气象的关联；')
nb.append("（ii）用神经网络学习$\\beta'(T,H,R)$并重建病例序列；")
nb.append('（iii）通过符号回归将NN黑箱翻译为显式解析公式。')
nb.append('')
nb.append('\\subsubsection{Phase 1: 传播效率学习}')
nb.append('')

content = content[:i1] + '\n'.join(nb) + content[i2e:]

# 2. Demote Phase 1 sub-subsections to paragraph
for old, new in [
    ('\\subsubsection{反推的$\\beta(t)$与气象的关系}', '\\paragraph{反推的$\\beta(t)$与气象的关系}'),
    ('\\subsubsection{NN拟合传播效率}', '\\paragraph{NN拟合传播效率}'),
    ('\\subsubsection{病例重建验证}', '\\paragraph{病例重建验证}'),
]:
    content = content.replace(old, new, 1)

# 3. Demote Phase 2 subsection -> subsubsection
content = content.replace(
    '\\subsection{Phase 2: 符号回归发现公式}',
    '\\subsubsection{Phase 2: 符号回归发现公式}', 1)

# 4. Wrap multi-city under Part II
old4 = '\\subsection{多城市验证}'
n4 = []
n4.append('% ------------------------------------------------------------')
n4.append('\\subsection{第二部分：多城市机制迁移与验证}')
n4.append('\\label{sec:part2}')
n4.append('% ------------------------------------------------------------')
n4.append('')
n4.append("本部分将第一部分在广州发现的$\\beta'(T,H,R)$公式\\textbf{不经重新训练}，")
n4.append('直接外推至广东其余15城市（共16城），检验机制的跨空间泛化能力。')
n4.append('评价重点从\u201c逐点贴合\u201d转向\u201c风险排序是否可靠、量级误差是否可控\u201d。')
n4.append('')
n4.append('\\subsubsection{多城市外推验证}')
content = content.replace(old4, '\n'.join(n4), 1)

# 5. Remove duplicated intro line
dup = "用广州训练的$\\beta'(T,H,R)$公式\\textbf{不经重新训练}，直接外推到其余15城市（共16城，2014年年度尺度）。\n"
content = content.replace(dup, '', 1)

# 6. Demote remaining subsections under Part II
for t in [
    '\\subsection{新旧数据结果对比（核心指标）}',
    '\\subsection{2014年暴发归因分析}',
    '\\subsection{扩展验证说明}',
]:
    content = content.replace(t, t.replace('\\subsection{', '\\subsubsection{'), 1)

for prefix in ['\\subsection{城市级月度指标分布', '\\subsection{时间窗口敏感性分析']:
    idx = content.find(prefix)
    if idx >= 0:
        end = content.find('}', idx) + 1
        old_h = content[idx:end]
        new_h = '\\subsubsection{' + old_h[len('\\subsection{'):]
        content = content[:idx] + new_h + content[end:]

with open('paper_draft/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("OK")
