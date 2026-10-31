r"""
Section 8 of Part III: the short-interval mean square of G_{1,q} split into its spectral diagonal and off-diagonal.

G = sum_i v_i with v_i the (mutually orthogonal) T_3-eigencomponents of G_{1,q} (Prony's method on (T_3^k G)(1), as in
newform_support.py of Part II).  For a window I = (X, X+H],
    S_total = sum_{n in I} |a_G(n)|^2,   S_diag = sum_i sum_{n in I} |a_{v_i}(n)|^2,   S_off = S_total - S_diag,
and the "aligned model" P(n) = sum_i sgn(a_{v_i}(N_0)) a_{v_i}(n), N_0 = X + H/2, which has the same diagonal and is
coherent at n = N_0.  The script prints, for dyadic H and sliding windows with X in [2^14, 2^15], the mean, rms and
extremes of S_off/S_diag for G and for the model, and saves the window trajectories for the figure.
Levels: q = 13, 17, 29, 37 (one to three newforms of level 4q plus old components) and q = 19, 23, 31, 43 (old
components only).  Floating point.  Usage: python short_interval_spectral.py [q ...]
"""
import os
import sys
import numpy as np
from math import pi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from sup_error_scan import r_table, sigma_over_n_odd   # noqa: E402
from local_densities import siegel_all                   # noqa: E402

qs = [int(a) for a in sys.argv[1:]] or [13, 17, 29, 37, 19, 23, 31, 43]
KMAX = 14
X0, X1 = 2 ** 14, 2 ** 15
XN = 2 ** 16
HS = [16, 64, 256, 1024, 4096, 16384]


def T3(G, n_max):
    n = np.arange(n_max + 1)
    H = np.zeros(n_max + 1)
    H[1:] = G[3 * n[1:]]
    H[3::3] += 3 * G[n[3::3] // 3]
    return H


def components(q):
    """T_3-eigencomponents of G_{1,q} on n <= XN (needs G up to 3^(dim-1) XN)."""
    Gmax = 3 ** KMAX + 1
    r = r_table(Gmax)
    fs, v2 = sigma_over_n_odd(Gmax)
    T, M = siegel_all(1, q, Gmax, r, fs, v2)
    G = T - M
    seqs = [G.copy()]
    for k in range(KMAX):
        seqs.append(T3(seqs[-1], len(seqs[-1]) // 3))
    s = np.array([v[1] for v in seqs])
    lam = None
    for dd in range(1, KMAX // 2 + 1):
        A = np.array([s[i:i + dd] for i in range(len(s) - dd)])
        coef = np.linalg.lstsq(A, s[dd:], rcond=None)[0]
        if np.abs(A @ coef - s[dd:]).max() < 1e-7 * np.abs(s).max():
            lam = np.sort(np.roots(np.r_[1, -coef[::-1]]).real)
            break
    if lam is None or len(lam) > 5:
        return None, None
    dim = len(lam)
    Yn = 3 ** (dim - 1) * XN + 1
    rn = r_table(Yn)
    fsn, v2n = sigma_over_n_odd(Yn)
    Tn, Mn = siegel_all(1, q, Yn, rn, fsn, v2n)
    Gn = Tn - Mn
    comps = []
    for i, li in enumerate(lam):
        v = Gn[: 3 ** (dim - 1) * XN + 1].copy()
        for j, lj in enumerate(lam):
            if j != i:
                v = T3(v, len(v) // 3) - lj * v[: len(v) // 3 + 1]
        v = v[: XN + 1] / np.prod([li - lj for j, lj in enumerate(lam) if j != i])
        if abs(v[1]) > 1e-9 * max(1, np.abs(v).max()):
            comps.append(v)
    return lam, comps


if __name__ == "__main__":
    out = []
    save = {}
    for q in qs:
        lam, comps = components(q)
        if comps is None:
            out.append(f"q = {q}: more than 5 T_3-eigenvalues, skipped")
            print(out[-1], flush=True)
            continue
        G = np.sum(comps, axis=0)
        k = len(comps)
        a2 = np.array([c ** 2 for c in comps])              # k x (XN+1)
        diag = a2.sum(axis=0)
        tot = G ** 2
        cum_tot = np.concatenate([[0.0], np.cumsum(tot)])
        cum_diag = np.concatenate([[0.0], np.cumsum(diag)])
        out.append(f"\nq = {q} (level {4 * q}): {k} orthogonal components, T_3-eigenvalues {np.round(lam, 4)};  "
                   f"whole range: S_off/S_diag = {(tot[1:].sum() - diag[1:].sum()) / diag[1:].sum():+.4f}")
        out.append(f"   {'H':>6} | {'G: mean':>8} {'rms':>7} {'min':>7} {'max':>7} | {'aligned model: mean':>19} {'max':>7}")
        for H in HS:
            Xs = np.arange(X0, X1 + 1, max(1, H // 8))
            St = cum_tot[Xs + H + 1] - cum_tot[Xs + 1]
            Sd = cum_diag[Xs + H + 1] - cum_diag[Xs + 1]
            ratio = (St - Sd) / Sd
            # aligned model: signs chosen at N0 = X + H/2 (component i gets sign of a_{v_i}(N0))
            N0 = Xs + H // 2
            sg = np.sign(np.array([c[N0] for c in comps]))       # k x len(Xs)
            sg[sg == 0] = 1
            # S_total^model = sum_{n in I} (sum_i sg_i a_i(n))^2 = S_diag + 2 sum_{i<j} sg_i sg_j sum_I a_i a_j
            cross = np.zeros(len(Xs))
            for i in range(k):
                for j in range(i + 1, k):
                    cij = np.concatenate([[0.0], np.cumsum(comps[i] * comps[j])])
                    cross += 2 * sg[i] * sg[j] * (cij[Xs + H + 1] - cij[Xs + 1])
            ratio_m = cross / Sd
            out.append(f"   {H:>6} | {ratio.mean():>+8.3f} {np.sqrt(np.mean(ratio ** 2)):>7.3f} {ratio.min():>+7.3f} {ratio.max():>+7.3f} | "
                       f"{ratio_m.mean():>+19.3f} {ratio_m.max():>+7.3f}")
            if H == 256:
                save[f"q{q}_X"] = Xs
                save[f"q{q}_tot"] = St
                save[f"q{q}_diag"] = Sd
        print("\n".join(out[-len(HS) - 2:]), flush=True)
    text = "\n".join(out)
    with open(os.path.join(HERE, '..', 'results', 'short_interval_spectral.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")
    np.savez(os.path.join(HERE, '..', 'results', 'short_interval_spectral.npz'), **save)
