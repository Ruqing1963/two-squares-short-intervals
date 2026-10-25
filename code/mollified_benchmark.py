r"""
Exploration B (Part III notes, Section 5): mollified theta weights and the sieve dimension.

With the weights w~(n) = r(n) (sum_{u | n, u <= N^xi} g_u)^2 r(N-n) (sum_{v | N-n, v <= N^xi} g_v)^2 the congruence sums
are combinations of T(m,m';N) with moduli d_1 [u_1,u_2] d_2 [v_1,v_2] <= d N^{4 xi}, so the level becomes
theta_eff = 1/2 - 4 xi (two-sided) or 1/2 - 2 xi (one-sided mollifier); a multiplicative g with (1 + g_p)^2 = 1/2 halves
the density at the primes p <= N^xi that it contains (notes, Proposition 5.2), so the sifting dimension becomes
kappa_0 = 1 (two-sided) or 3/2 (one-sided) for z <= N^xi, while the Richert range (N^xi, N^b] keeps the density 2 of
r(n) r(N-n).  For xi < a the mollifier is identically 1 on the sifted integers and the sieve is unchanged at a lower
level (Theorem 5.1).  This script scans the hybrid sieve for xi >= a:
    L = f_{kappa_0}(theta_eff / a),
    R = int_a^b (1 - t/b) kappa(t) F_{kappa_0}((theta_eff - t)/a) dt/t,   kappa(t) = kappa_0 (t < xi), 2 (t > xi),
and prints the smallest attainable 1/lam + 2/b = R/L + 2/b and R/L + 1/b against the unmollified 7.56 / 5.07 of
Part II (theta = 1/2, kappa = 2).  kappa_0 = 3/2 needs the DHR constants alpha, beta for kappa = 3/2, which are found
by the solver of dhr_sieve.py from a rough guess.  Floating point.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from dhr_sieve import DHR   # noqa: E402


def LR(D, k0, theta, a, xi, b, nu=400):
    L = float(D.fs(theta / a))
    t = np.linspace(a, b, nu + 1)
    kap = np.where(t < xi, k0, 2.0)
    R = float(np.trapz((1 - t / b) * kap * D.Fs((theta - t) / a) / t, t))
    return L, R


def scan(D, k0, sides, label, out):
    best_j = best_s = None
    for xi in np.arange(0.02, 0.125, 0.0025):
        theta = 0.5 - 2 * sides * xi
        if theta <= 0:
            continue
        for a in np.arange(0.01, min(xi, theta / D.beta) + 1e-9, 0.0025):
            for b in np.arange(a + 0.02, theta - 0.004, 0.005):
                L, R = LR(D, k0, theta, a, xi, b)
                if L <= 0:
                    continue
                jt, sd = R / L + 2 / b, R / L + 1 / b
                if best_j is None or jt < best_j[0]:
                    best_j = (jt, sd, xi, theta, a, b, L, R)
                if best_s is None or sd < best_s[1]:
                    best_s = (jt, sd, xi, theta, a, b, L, R)
    for tag, r in (("best joint", best_j), ("best side ", best_s)):
        out.append(f"   {label}: {tag}: 1/lam+2/b = {r[0]:.3f}, 1/lam+1/b = {r[1]:.3f} at xi = {r[2]:.4f} "
                   f"(theta_eff = {r[3]:.3f}), a = {r[4]:.4f}, b = {r[5]:.3f}, L = {r[6]:.4f}, R = {r[7]:.4f}")


if __name__ == "__main__":
    out = ["unmollified (Part II, theta = 1/2, kappa = 2): joint 7.561, side 5.065"]
    D1 = DHR(1, guess=(2.0, 2.0), verbose=False)
    scan(D1, 1.0, 2, "two-sided mollifier, kappa_0 = 1, theta_eff = 1/2 - 4 xi", out)
    try:
        D15 = DHR(1.5, guess=(4.2, 3.3), verbose=True)
        ok = abs(float(D15.Fs(40.0)) - 1) < 1e-3 and abs(float(D15.fs(40.0)) - 1) < 1e-3
        out.append(f"kappa = 3/2: alpha = {D15.alpha:.4f}, beta = {D15.beta:.4f}, F(40) = {float(D15.Fs(40.0)):.5f}, "
                   f"f(40) = {float(D15.fs(40.0)):.5f}  ({'accepted' if ok else 'NOT converged; one-sided case skipped'})")
        if ok:
            scan(D15, 1.5, 1, "one-sided mollifier, kappa_0 = 3/2, theta_eff = 1/2 - 2 xi", out)
    except Exception as exc:  # noqa: BLE001
        out.append(f"kappa = 3/2 solver failed: {exc!r}; one-sided case skipped")
    # for comparison: the same hybrid formula with no level cost (what the mollifier would give if it were free)
    D2 = DHR(2, verbose=False)
    out.append("for comparison, the hypothetical case without level cost (theta_eff = 1/2):")
    best = None
    for xi in np.arange(0.05, 0.45, 0.01):
        for a in np.arange(0.02, min(xi, 0.5 / 2) - 1e-9, 0.0025):
            for b in np.arange(a + 0.02, 0.496, 0.005):
                L, R = LR(D1, 1.0, 0.5, a, xi, b)
                if L <= 0:
                    continue
                jt = R / L + 2 / b
                if best is None or jt < best[0]:
                    best = (jt, R / L + 1 / b, xi, a, b)
    out.append(f"   two-sided, kappa_0 = 1 below N^xi, 2 above, level 1/2 kept: best joint {best[0]:.3f} (side {best[1]:.3f}) "
               f"at xi = {best[2]:.2f}, a = {best[3]:.4f}, b = {best[4]:.3f}")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, '..', 'results', 'mollified_benchmark.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")
