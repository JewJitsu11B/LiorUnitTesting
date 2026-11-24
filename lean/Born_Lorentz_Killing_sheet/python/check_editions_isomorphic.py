"""Structural comparison of the matrix and Clifford editions.

Checks that the two .tex files have the same skeleton: the same sequence of blocks
(boxes, headings, display equations, group-class notes, chain rows) and the same
machine-check tag after each one. It does not check that the mathematics of a line
is the translation of its partner; that is the job of check_clifford_dictionary.py.
"""
import re
import sys

A, B = sys.argv[1], sys.argv[2]
BS = chr(92)


def skeleton(path):
    src = open(path, encoding="utf-8").read()
    src = src[src.index(BS + "begin{document}"):]
    pat = re.compile(
        "(" + "|".join([
            re.escape(BS + "begin{sheetbox}"), re.escape(BS + "begin{smallbox}"),
            re.escape(BS + "begin{longtable}"), re.escape(BS + "begin{tikzcd}"),
            re.escape(BS + "begin{tabular}"), re.escape(BS + "repline"),
            re.escape(BS + "forcesline"), re.escape(BS + "midrule"),
            re.escape(BS + "[") + r"\n", re.escape("{" + BS + "large" + BS + "textbf{"),
            re.escape(BS + "textbf{") + r"[A-Z][^}]*\}", re.escape(BS + "tg{"),
        ]) + ")")
    out = []
    for m in pat.finditer(src):
        tok = m.group(1)
        if tok == BS + "tg{":
            depth, i = 1, m.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(src[i], 0)
                i += 1
            out.append("TAG " + src[m.end():i - 1])
        elif tok.startswith(BS + "textbf{"):
            out.append("BOLD")
        else:
            out.append(tok.strip())
    return out


sa, sb = skeleton(A), skeleton(B)
print("blocks:", len(sa), len(sb))
bad = 0
for i in range(max(len(sa), len(sb))):
    x = sa[i] if i < len(sa) else None
    y = sb[i] if i < len(sb) else None
    if x != y:
        bad += 1
        if bad <= 15:
            print("DIFF at", i, "|", x, "|", y)
tags = sum(1 for t in sa if t.startswith("TAG"))
print("tags:", tags, "| mismatches:", bad)
sys.exit(1 if bad else 0)
