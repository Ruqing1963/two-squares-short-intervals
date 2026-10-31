r"""
Figure 4 of Part III (Sections 7-8): (a) the maximal window ratio max_X rho(X,H) of the short-interval mean square of
G_{m,m'} against H for four levels L (results/short_interval_mean_square.txt), with the inflation 1 + L/H that a
term L in the bound would allow; (b) the spectral diagonal S_diag and the total S_total of the short-interval mean
square of G_{1,29} along sliding windows of length 256 (results/short_interval_spectral.npz).
Usage: python plot_short_interval.py
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#e1e0d9"

blocks = {}
cur = None
for line in open(os.path.join(RES, 'short_interval_mean_square.txt'), encoding='utf-8'):
    m = re.match(r"\(m,m'\) = \((\d+),(\d+)\), L = (\d+)", line)
    if m:
        cur = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        blocks[cur] = []
        continue
    m = re.match(r"\s+(\d+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|", line)
    if m and cur is not None:
        blocks[cur].append((int(m.group(1)), float(m.group(4))))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
cols = {20: AQUA, 1508: ORANGE, 40028: BLUE, 41612: INK}
for key, rows in sorted(blocks.items(), key=lambda kv: kv[0][2]):
    L = key[2]
    if L not in cols:
        continue
    H = np.array([r[0] for r in rows], dtype=float)
    mx = np.array([r[1] for r in rows])
    ax.plot(H, mx, color=cols[L], lw=1.8, marker="o", ms=3, label=f"$({key[0]},{key[1]})$, $L={L}$: $\\max_X\\rho(X,H)$")
    ax.plot(H, 1 + L / H, color=cols[L], lw=1, ls="--")
ax.plot([], [], color=MUTED, lw=1, ls="--", label=r"$1+L/H$ (what a term $L$ would allow)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("window length $H$")
ax.set_ylabel(r"$\max_X\rho(X,H)$, $X\in[2^{19},2^{20}]$")
ax.set_title("(a) short-interval mean square of $G_{m,m'}$: no inflation with $L$", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

dat = np.load(os.path.join(RES, 'short_interval_spectral.npz'))
q = 29
X, tot, diag = dat[f"q{q}_X"], dat[f"q{q}_tot"], dat[f"q{q}_diag"]
scale = diag.mean()
bx.plot(X, tot / scale, color=BLUE, lw=1.2, label=r"$S_{\rm total}(X,H)$")
bx.plot(X, diag / scale, color=ORANGE, lw=1.2, label=r"$S_{\rm diag}(X,H)$")
bx.axhline(1, color=MUTED, lw=0.8)
ratio = (tot - diag) / diag
bx.text(0.02, 0.95, f"$S_{{\\rm off}}/S_{{\\rm diag}}$: mean {ratio.mean():+.3f}, rms {np.sqrt(np.mean(ratio ** 2)):.3f}, "
        f"range [{ratio.min():+.2f}, {ratio.max():+.2f}]", transform=bx.transAxes, fontsize=7.5, va="top")
bx.set_xlabel("window start $X$")
bx.set_ylabel("window sum / mean diagonal")
bx.set_title(f"(b) spectral diagonal and total, $G_{{1,{q}}}$ (level ${4 * q}$), $H=256$", fontsize=10)
bx.legend(fontsize=8, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "short_interval.pdf"))
fig.savefig(os.path.join(FIG, "short_interval.png"), dpi=160)
print("written figures/short_interval.pdf/.png")
