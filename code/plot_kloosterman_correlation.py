r"""
Figure 2 of Part III (Section 4): (a) the correlation matrix C(c1,c2)/sqrt(C(c1,c1)C(c2,c2)) of the h-sums of
Kloosterman sums S(16N, m|h|^2; m'k) for k = 127 and twelve primes m' (results/kloosterman_correlation_k127.npz);
(b) the partial quadratic forms sum_{j,l<=M} s_j s_l C_{jl} / sum_{j<=M} C_{jj} against M for coherent signs (all +1)
and random signs (mean and one standard deviation over 200 draws), for k = 43, 83, 127.
Usage: python plot_kloosterman_correlation.py
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
rng = np.random.default_rng(11)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
dat = np.load(os.path.join(RES, 'kloosterman_correlation_k127.npz'))
M, mps = dat['M'], dat['mps']
d = np.sqrt(np.diag(M))
corr = M / np.outer(d, d)
show = corr.copy()
np.fill_diagonal(show, np.nan)
im = ax.imshow(show, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(len(mps)))
ax.set_yticks(range(len(mps)))
ax.set_xticklabels([str(int(p)) for p in mps], fontsize=7.5, rotation=90)
ax.set_yticklabels([str(int(p)) for p in mps], fontsize=7.5)
ax.set_xlabel("$m'_2$")
ax.set_ylabel("$m'_1$")
ax.set_title("(a) correlations of the $h$-sums, $k=127$, $N=2000002$, $m=5$", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=7.5)
off = corr[~np.eye(len(mps), dtype=bool)]
ax.text(0.02, -0.22, f"off-diagonal rms {np.sqrt(np.mean(off ** 2)):.3f}, max $|\\cdot|$ {np.abs(off).max():.3f}; "
        f"random signs on $\\asymp H^2$ points: {1 / np.sqrt(600):.3f}", transform=ax.transAxes, fontsize=7.5)

for k, col in ((43, AQUA), (83, ORANGE), (127, BLUE)):
    dat = np.load(os.path.join(RES, f'kloosterman_correlation_k{k}.npz'))
    M = dat['M']
    n = M.shape[0]
    Ms = np.arange(1, n + 1)
    coh = [np.ones(mm) @ M[:mm, :mm] @ np.ones(mm) / np.trace(M[:mm, :mm]) for mm in Ms]
    rnd = np.array([[(s[:mm] @ M[:mm, :mm] @ s[:mm]) / np.trace(M[:mm, :mm]) for mm in Ms]
                    for s in rng.choice([-1.0, 1.0], size=(200, n))])
    bx.plot(Ms, coh, color=col, lw=1.8, marker="o", ms=3, label=f"all $+1$, $k={k}$")
    bx.plot(Ms, rnd.mean(axis=0), color=col, lw=1.2, ls="--", label=f"random signs (mean $\\pm$ sd), $k={k}$")
    bx.fill_between(Ms, rnd.mean(axis=0) - rnd.std(axis=0), rnd.mean(axis=0) + rnd.std(axis=0), color=col, alpha=0.12, lw=0)
bx.plot(Ms, Ms, color=MUTED, lw=0.8, ls=":")
bx.text(8.3, 8.9, "full coherence ($=M$)", color=MUTED, fontsize=7.5, rotation=38)
bx.axhline(1, color=MUTED, lw=0.8)
bx.set_xlabel("number $M$ of moduli $m'$ (in increasing order)")
bx.set_ylabel("quadratic form / trace")
bx.set_title(r"(b) $\sum_{j,l\leq M}\sigma_j\sigma_l C_{jl}\,/\,\sum_{j\leq M}C_{jj}$ against $M$", fontsize=10)
bx.set_ylim(0, 3.6)
bx.legend(fontsize=7, loc="upper left", framealpha=0.95, ncol=2)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.9, wspace=0.35)
fig.savefig(os.path.join(FIG, "kloosterman_correlation.pdf"))
fig.savefig(os.path.join(FIG, "kloosterman_correlation.png"), dpi=160)
print("written figures/kloosterman_correlation.pdf/.png")
