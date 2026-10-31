r"""
Figure 1 of Part III (Section 2): (a) sum_d |R_d| / sum |R(d1,d2)| against theta, with the random-sign values 1/sqrt2
and 1/2 (results/symmetric_remainder_curves.json); (b) the share of the joint (+1)-eigenspace of the Atkin-Lehner
involutions in G_{1,d}: Rankin-Selberg norm share against the share of the sieve coefficients a(2n') - a(n'), for the
moduli of results/symmetric_norm_share.txt, with the equidistribution values 1/2, 1/4, 1/8 and the empirical 0.424.
Usage: python plot_symmetric_remainder.py
"""
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#e1e0d9"

curves = json.load(open(os.path.join(RES, 'symmetric_remainder_curves.json')))
series = {}
for key, val in curves.items():
    th, om = key.split(',')
    series.setdefault(int(om), []).append((float(th), val[0], val[1], val[2]))
for om in series:
    series[om].sort()

shares = []   # (d, omega, chi, norm share of all-plus, sieve share of all-plus)
cur = None
for line in open(os.path.join(RES, 'symmetric_norm_share.txt'), encoding='utf-8'):
    m = re.match(r"d = (\d+) = ([\d*]+) \(chi\(d\) = ([+-]1)\): norm shares (.*?);", line)
    if m:
        d, chi = int(m.group(1)), int(m.group(3))
        pats = dict(re.findall(r"\(([+-]+)\) ([\d.]+)", m.group(4)))
        cur = [d, len(m.group(2).split('*')), chi, pats, None]
        continue
    m = re.match(r"\s+shares over\s+a\(2n'\)-a\(n'\): (.*)", line)
    if m and cur is not None:
        pats = dict(re.findall(r"\(([+-]+)\) ([\d.]+)", m.group(1)))
        cur[4] = pats
        shares.append(cur)
        cur = None

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
for om, col, lab, ref in ((2, BLUE, r"$\omega(d)=2$", 1 / np.sqrt(2)), (3, ORANGE, r"$\omega(d)=3$", 0.5), (0, AQUA, "all $d$", None)):
    s = np.array(series[om])
    ax.errorbar(s[:, 0], s[:, 1], yerr=s[:, 2], color=col, lw=1.8, marker="o", ms=3.5, capsize=2, label=lab)
    if ref is not None:
        ax.axhline(ref, color=col, lw=1, ls="--", label=f"random signs: {ref:.3f}")
ax.set_xlabel(r"level $\theta$ ($d\leq N^\theta$)")
ax.set_ylabel(r"$\sum_d|R_d|\,/\,\sum_{d_1d_2=d}|R(d_1,d_2;N)|$")
ax.set_title("(a) the symmetric remainder against the absolute sum (24 values of $N$)", fontsize=10)
ax.set_xlim(0.39, 0.76)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# (b)
for om, mk, ref in ((2, "o", 0.5), (3, "s", 0.25), (4, "^", 0.125)):
    pts = [s for s in shares if s[1] == om and s[2] == 1 and s[4] is not None]
    if not pts:
        continue
    key = "+" * om
    d = np.array([p[0] for p in pts], dtype=float)
    norm = np.array([float(p[3][key]) for p in pts])
    sieve = np.array([float(p[4][key]) for p in pts])
    bx.scatter(d, norm, color=BLUE, marker=mk, s=28, label=f"norm share, $\\omega(d)={om}$" if om == 2 else None)
    bx.scatter(d, sieve, color=ORANGE, marker=mk, s=28, label=f"sieve share, $\\omega(d)={om}$" if om == 2 else None)
    bx.axhline(ref, color=MUTED, lw=0.8, ls=":")
    bx.text(13, ref + 0.012, f"equidistribution $1/{2 ** (om - 1)}$", color=MUTED, fontsize=7.5)
bx.axhline(0.424, color=ORANGE, lw=1, ls="--")
bx.text(13, 0.434, "empirical share of the remainders, 0.424", color=ORANGE, fontsize=7.5)
bx.scatter([], [], color=MUTED, marker="s", s=28, label=r"$\omega(d)=3$ (squares), $4$ (triangles)")
bx.set_xscale("log")
bx.set_xlabel("$d$ (all prime factors $\\equiv 1$ mod 4)")
bx.set_ylabel("share of the all-$(+)$ eigenspace")
bx.set_title("(b) all-$(+)$ Atkin--Lehner eigenspace of $G_{1,d}$: norm and sieve shares", fontsize=10)
bx.set_ylim(0, 0.7)
bx.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "symmetric_remainder.pdf"))
fig.savefig(os.path.join(FIG, "symmetric_remainder.png"), dpi=160)
print("written figures/symmetric_remainder.pdf/.png")
