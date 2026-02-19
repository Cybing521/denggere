#!/usr/bin/env python3
import re, os
from docx import Document

os.chdir('/root/wenmei')
filepath = 'paper_biye/thesis_draft.tex'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# PART 1: Convert all list environments to flowing prose
list_count = [0]

def list_replacer(match):
    list_count[0] += 1
    body = match.group(2)
    parts = re.split(r'\\item\s+', body)
    items = [p for p in parts if p.strip()]
    has_eq = any('\\begin{equation}' in it for it in items)
    LP = '\uff08'
    RP = '\uff09'
    result_parts = []
    for i, item in enumerate(items, 1):
        item = item.strip()
        if '\\begin{equation}' in item:
            before, rest = item.split('\\begin{equation}', 1)
            eq_body, after = rest.split('\\end{equation}', 1)
            before = re.sub(r'\s*\n\s*', ' ', before).strip()
            after = re.sub(r'\s*\n\s*', ' ', after).strip()
            entry = LP + str(i) + RP + before + '\n\\begin{equation}\n' + eq_body.strip() + '\n\\end{equation}'
            if after:
                entry += '\n' + after
            result_parts.append(entry)
        else:
            item = re.sub(r'\s*\n\s*', ' ', item)
            result_parts.append(LP + str(i) + RP + item)
    if has_eq:
        return '\n\n'.join(result_parts)
    else:
        return ''.join(result_parts)

content = re.sub(
    r'\\begin\{(enumerate|itemize)\}\[leftmargin=2em\]\s*\n(.*?)\\end\{\1\}',
    list_replacer, content, flags=re.DOTALL)
print(f"[PART 1] Converted {list_count[0]} list environments to prose")

# PART 2: Replace bibitems with correct docx references
def extract_refs(docx_path):
    refs = {}
    for p in Document(docx_path).paragraphs:
        t = p.text.strip()
        if t.startswith('['):
            m = re.match(r'\[(\d+)\]\s*(.*)', t)
            if m:
                refs[int(m.group(1))] = m.group(2).strip()
    return refs

qy = extract_refs('paper_biye/\u524d\u8a00.docx')
p1 = extract_refs('paper_biye/\u7b2c\u4e00\u90e8\u5206.docx')
print(f"[PART 2] Extracted {len(qy)} refs from qianyan, {len(p1)} refs from part1")

key_map = {
    'messina2019': (qy, 1), 'bhatt2013': (qy, 2), 'who2024': (qy, 3),
    'yue2021': (qy, 4), 'lai2015': (qy, 5), 'cheng2016': (qy, 6),
    'liyanage2016': (qy, 7), 'desouza2024': (qy, 8), 'shapiro2017': (qy, 9),
    'lambrechts2011': (qy, 10), 'kamiya2020': (qy, 11), 'mordecai2019': (qy, 12),
    'colon2018': (qy, 13), 'zhou2025': (qy, 14), 'nosrat2021': (qy, 15),
    'roiz2015': (qy, 16), 'chengq2023': (qy, 17), 'polrob2025': (qy, 18),
    'wu2018': (qy, 19), 'dacosta2025': (qy, 20), 'leung2023': (qy, 21),
    'liuk2020': (qy, 22), 'sehi2025': (qy, 23), 'luo2025': (qy, 24),
    'chengy2025': (qy, 25), 'baker2022': (qy, 26), 'mills2024': (qy, 27),
    'ahman2025': (qy, 28), 'smith2012': (qy, 29), 'guo2024': (qy, 30),
    'zhu2016': (qy, 31), 'liuy2023': (qy, 32), 'din2021': (qy, 33),
    'mordecai2017': (qy, 34), 'chen2024science': (qy, 35), 'caldwell2021': (qy, 36),
    'li2019pnas': (qy, 37), 'zhangs2021': (qy, 38), 'yang2023': (qy, 39),
    'lir2024': (qy, 40), 'nikparvar2021': (qy, 41), 'murphy2021': (qy, 42),
    'holm2019': (qy, 43), 'kamyshnyi2026': (qy, 44), 'adeoye2025': (qy, 45),
    'makke2024': (qy, 46), 'fajardo2024': (qy, 47), 'zhang2024plos': (qy, 48),
    'ouedraogo2025': (p1, 1), 'white2025': (p1, 3), 'huber2018': (p1, 4),
    'lic2023': (p1, 5), 'dennington2025': (p1, 6), 'chengj2021': (p1, 8),
}
keep_keys = {'ccm14', 'chan2012', 'brady2013', 'kingma2015', 'cranmer2023',
             'kraemer2019', 'chen2018node', 'lowe2021', 'roberts2017'}

def latex_escape(text):
    text = text.replace('&', '\\&')
    text = text.replace('\u2013', '--')
    text = text.replace('\u00b0', '$^\\circ$')
    def url_fix(m):
        url = m.group(0)
        trail = ''
        while url and url[-1] in '.,;:':
            trail = url[-1] + trail
            url = url[:-1]
        return '\\url{' + url + '}' + trail
    text = re.sub(r'https?://\S+', url_fix, text)
    return text

bib_tag = '\\begin{thebibliography}{99}'
bib_end_tag = '\\end{thebibliography}'
bib_start = content.index(bib_tag)
bib_end = content.index(bib_end_tag) + len(bib_end_tag)
bib_section = content[bib_start:bib_end]

positions = [(m.start(), m.end(), m.group(1))
             for m in re.finditer(r'\\bibitem\{([^}]+)\}', bib_section)]
parsed = []
for j, (start, end, key) in enumerate(positions):
    if j + 1 < len(positions):
        text_end = positions[j + 1][0]
    else:
        text_end = bib_section.index('\\end{thebibliography}')
    text = bib_section[end:text_end].strip()
    parsed.append((key, text))
print(f"[PART 2] Parsed {len(parsed)} existing bibitems")

new_entries = []
replaced = 0
for key, old_text in parsed:
    if key in key_map:
        src, num = key_map[key]
        if num in src:
            new_text = latex_escape(src[num])
            new_entries.append('\\bibitem{' + key + '} ' + new_text)
            replaced += 1
        else:
            print(f"  WARNING: ref #{num} not found for key '{key}'")
            new_entries.append('\\bibitem{' + key + '} ' + old_text)
    elif key in keep_keys:
        new_entries.append('\\bibitem{' + key + '} ' + old_text)
    else:
        print(f"  WARNING: unknown key '{key}', keeping as-is")
        new_entries.append('\\bibitem{' + key + '} ' + old_text)

print(f"[PART 2] Replaced {replaced}/{len(parsed)} bibitems")

new_bib = ('\\begin{thebibliography}{99}\n\\small\n\n'
           + '\n\n'.join(new_entries)
           + '\n\n\\end{thebibliography}')
content = content[:bib_start] + new_bib + content[bib_end:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"\nDone! Written to {filepath}")
