#!/usr/bin/env python3
import os,glob
H=os.path.dirname(os.path.abspath(__file__))
P=sorted(glob.glob(os.path.join(H,"tex_parts","*.tex")))
O=os.path.join(H,"thesis_draft.tex")
with open(O,"w") as f:
 for p in P: f.write(open(p).read()+chr(10))
print("OK",os.path.getsize(O),"bytes")
