r"""
Numerical test of the short-interval mean square (S_Delta) of the notes, Section 7:
    sum_{X < n <= X+H} |a_G(n)|^2  <<  (H + L) X ||G||^2 / L          (G = G_{m,m'}, L = 4 m m').
The Rankin-Selberg mean of |a_G(n)|^2 at n is c(n) = 16 pi^2 ||G||^2 n / vol(Gamma_0(L)\H), so the natural
normalisations are
    rho_mean(X,H) = sum_{X<n<=X+H} |a(n)|^2 / sum_{X<n<=X+H} c(n)              (1 if the mass is spread),
    rho_S(X,H)    = rho_mean / (1 + L/H)                                        (the (S_Delta) normalisation).
For each pair (m,m') the coefficients a(n) = T - M are computed exactly for all n <= 2^21 (local_densities.siegel_all),
||G||^2 is the Rankin-Selberg estimate at 2^21, and for dyadic H the script prints the mean and the maximum of
rho_mean over all windows (X, X+H] with X in [2^19, 2^20] (sliding window via cumulative sums), for all n and for the
admissible n only (n = 2 mod 4, (n, mm') = 1, the N the sieve uses).  rho_max >> 1 for H near L would mean that the
coefficient mass concentrates; rho_max = O(1) down to H ~ L is the content of (S_Delta).  Floating point.
Usage: python short_interval_mean_square.py [m,m' ...]
"""
import os
import sys
import time
import numpy as np
from math import pi, gcd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from sup_error_scan import r_table, sigma_over_n_odd, factor   # noqa: E402
from local_densities import siegel_all                          # noqa: E402

pairs = [tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]] or \
        [(1, 5), (3, 5), (1, 101), (13, 29), (1, 1009), (1, 10007), (101, 103)]
XMAX = 2 ** 21
X0, X1 = 2 ** 19, 2 ** 20
HS = [2 ** j for j in range(4, 20)]


def vol(L):
    v = pi / 3 * L
    for p in factor(L):
        v *= 1 + 1.0 / p
    return v


if __name__ == "__main__":
    t0 = time.time()
    r = r_table(XMAX)
    fs, v2 = sigma_over_n_odd(XMAX)
    n = np.arange(XMAX + 1, dtype=float)
    out = [f"windows (X, X+H], X in [{X0}, {X1}]; rho_mean = sum_I |a|^2 / sum_I c(n), c(n) = 16 pi^2 ||G||^2 n / vol;  "
           f"rho_S = rho_mean / (1 + L/H)"]
    for (m, mp) in pairs:
        L = 4 * m * mp
        T, M = siegel_all(m, mp, XMAX, r, fs, v2)
        a = T - M
        a[0] = 0.0
        V = vol(L)
        normG = np.sum(a[1:] ** 2 / n[1:]) * V / (16 * pi ** 2 * XMAX)
        c = 16 * pi ** 2 * normG * n / V
        adm = np.zeros(XMAX + 1, dtype=bool)
        adm[2::4] = True
        for p in factor(m * mp):
            adm[::p] = False
        a2 = a ** 2
        cum = np.concatenate([[0.0], np.cumsum(a2)])
        cumc = np.concatenate([[0.0], np.cumsum(c)])
        cum_adm = np.concatenate([[0.0], np.cumsum(a2 * adm)])
        cumc_adm = np.concatenate([[0.0], np.cumsum(c * adm)])
        out.append(f"\n(m,m') = ({m},{mp}), L = {L}, ||G||^2 (RS at 2^21) = {normG:.4f}, mean |a|^2 at n = 2^19.5: "
                   f"{16 * pi ** 2 * normG * 2 ** 19.5 / V:.1f}   [{time.time() - t0:.0f}s]")
        out.append(f"   {'H':>7} {'H/L':>9} | {'all n: mean':>11} {'max':>7} {'rho_S max':>9} | {'admissible: mean':>16} {'max':>7} {'rho_S max':>9}")
        for H in HS:
            Xs = np.arange(X0, X1 + 1, max(1, H // 8))
            S = cum[Xs + H + 1] - cum[Xs + 1]
            C = cumc[Xs + H + 1] - cumc[Xs + 1]
            rho = S / C
            Sa = cum_adm[Xs + H + 1] - cum_adm[Xs + 1]
            Ca = cumc_adm[Xs + H + 1] - cumc_adm[Xs + 1]
            ok = Ca > 0
            rhoa = Sa[ok] / Ca[ok]
            fac = 1 / (1 + L / H)
            out.append(f"   {H:>7} {H / L:>9.3f} | {rho.mean():>11.3f} {rho.max():>7.3f} {rho.max() * fac:>9.3f} | "
                       f"{rhoa.mean():>16.3f} {rhoa.max():>7.3f} {rhoa.max() * fac:>9.3f}")
        print("\n".join(out[-len(HS) - 2:]), flush=True)
    text = "\n".join(out)
    with open(os.path.join(HERE, '..', 'results', 'short_interval_mean_square.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")
