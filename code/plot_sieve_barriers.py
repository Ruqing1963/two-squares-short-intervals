r"""
Figure 3 of Part III (Sections 5-6): (a) the best thresholds 1/lam + 2/b of the hybrid mollified sieves against the
mollifier length xi (results/mollified_curves.txt), with the unmollified kappa = 2 value 7.561 and the hypothetical
kappa = 1 value 5.181 at theta = 1/2; (b) the LP-optimal weight g*(t) against the linear weight lam(1 - t/b*) for the
pairs (7,5), (7,4), (6,4) (results/optimal_weights_*.npz), with the five-prime configuration that binds (7,4).
Usage: python plot_sieve_barriers.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#e1e0d9"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
rows = [l.split() for l in open(os.path.join(RES, 'mollified_curves.txt')) if l[0] != 'v']
for tag, col, lab in (("two-sided", BLUE, r"two-sided, $\kappa_0=1$, $\theta_{\rm eff}=1/2-4\xi$"),
                      ("one-sided", ORANGE, r"one-sided, $\kappa_0=3/2$, $\theta_{\rm eff}=1/2-2\xi$")):
    sel = [r for r in rows if r[0].startswith(tag)]
    if not sel:
        continue
    xi = np.array([float(r[1]) for r in sel])
    jt = np.array([float(r[3]) for r in sel])
    sd = np.array([float(r[4]) for r in sel])
    ax.plot(xi, jt, color=col, lw=1.8, marker="o", ms=3, label=lab + r": $\min(1/\lambda+2/b)$")
    ax.plot(xi, sd, color=col, lw=1.2, ls="--", label=lab.split(',')[0] + r": $\min(1/\lambda+1/b)$")
ax.axhline(7.561, color=INK, lw=1, ls=":")
ax.text(0.021, 7.63, r"unmollified, $\kappa=2$, $\theta=1/2$: 7.561 ($(7,5)$)", fontsize=7.5)
ax.axhline(5.181, color=AQUA, lw=1, ls=":", label=r"hypothetical $\kappa=1$ at $\theta=1/2$: 5.181 ($(5,3)$)")
ax.axhline(5.065, color=INK, lw=0.8, ls=":")
ax.text(0.07, 4.7, r"unmollified $\min(1/\lambda+1/b)=5.065$", fontsize=7.5)
ax.set_xlabel(r"mollifier length $\xi$ ($u\leq N^\xi$)")
ax.set_ylabel("threshold")
ax.set_title("(a) hybrid mollified sieves: the level lost against the dimension gained", fontsize=10)
ax.set_ylim(3.8, 16)
ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

for pair, col in (("75", BLUE), ("74", ORANGE), ("64", AQUA)):
    dat = np.load(os.path.join(RES, f'optimal_weights_{pair}.npz'))
    t, g, a, b = dat['t'], dat['g'], float(dat['a']), float(dat['b_linear'])
    k, kp = int(pair[0]), int(pair[1])
    lam = 1 / min(k + 1 - 2 / b, kp + 1 - 1 / b)
    lin = np.maximum(0.0, lam * (1 - t / b))
    bx.plot(t, lin, color=col, lw=1.2, ls="--")
    bx.plot(t[::4], g[::4], color=col, lw=0, marker="o", ms=2.6, label=f"$({k},{kp})$: LP optimum (dots), $\\lambda(1-t/b^*)$ (dashed), $a={a:.4f}$, $b^*={b:.4f}$")
    if pair == "74":
        conf = dat['conf5']
        gi = np.interp(conf, t, g)
        bx.plot(conf, gi, color=INK, lw=0, marker="x", ms=7, mew=1.5)
        bx.annotate("five-prime configuration\nattaining $G_5(1)=1$ for $(7,4)$:\n$(0.115,0.115,0.115,0.318,0.338)$",
                    (conf[-1], gi[-1]), xytext=(0.30, 0.30), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
bx.set_xlabel(r"$t=\log p/\log N$")
bx.set_ylabel(r"weight $g(t)$ (normalised to $1/\lambda=1$)")
bx.set_title("(b) the LP-optimal weight is the linear weight", fontsize=10)
bx.set_xlim(0.04, 0.5)
bx.set_ylim(0, 0.45)
bx.legend(fontsize=7, loc="upper right", framealpha=0.95)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sieve_barriers.pdf"))
fig.savefig(os.path.join(FIG, "sieve_barriers.png"), dpi=160)
print("written figures/sieve_barriers.pdf/.png")
