import os, re, glob
DIR = os.path.dirname(os.path.abspath(__file__))
parts = []
for p in sorted(glob.glob(os.path.join(DIR, '_part*.tex'))):
    with open(p, 'r', encoding='utf-8') as f:
        parts.append(f.read())
out = os.path.join(DIR, 'thesis_draft.tex')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(parts))
txt = '\n'.join(parts)
clean = re.sub(r'\\[a-zA-Z]+', '', txt)
clean = re.sub(r'[{}\[\]\\$&%#_^~\n\r\t ]', '', clean)
print(f"Parts: {len(parts)}, Chars: {len(clean)}, Bibitems: {txt.count('bibitem')}")
lean = re.sub(r'\\[a-zA-Z]+', '', txt)
clean = re.sub(r'[{}\[\]\\$&%#_^~\n\r\t ]', '', clean)
print(f"Approx chars: {len(clean)}")
print(f"Bibitem count: {txt.count(chr(92)+'bibitem')}")
