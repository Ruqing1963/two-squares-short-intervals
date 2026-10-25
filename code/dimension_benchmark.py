r"""
Research 3 (Part III notes, Section 3): the price of the weights r(n) r(N-n) is the sieve dimension, not the factor 2^Omega.

Under the weight w(n) = r(n) r(N-n) the density of p | n(N-n) is 2h(p) ~ 4/p for split p (dimension kappa = 2); under
the indicator weight b(n) b(N-n) (n and N-n sums of two squares) it would be ~ 2/p for split p and ~ 2/q^2 for inert q
(dimension kappa = 1, the dimension of Goldbach's problem).  The factor 2^Omega(n(N-n)) = w(n)/16 enters the final count
only through the constant 1/max w; the pairs (k, k') come from the thresholds 1/lam + 2/b and 1/lam + 1/b of the
Richert weights, which depend on kappa and on the level theta.  This script computes, for both dimensions and several
levels theta, the smallest attainable thresholds
    min (R/L + 2/b),  min (R/L + 1/b),   L = f_kappa(theta/a),  R = kappa int_a^b (1 - t/b) F_kappa((theta - t)/a) dt/t,
over the sifting parameter a and the weight cut b (0 < a < b < theta, a < theta/beta_kappa), i.e. the pairs (k, k') that
the same Richert-weight machinery would give if the remainder sums for the indicator weights had the level theta.
kappa = 1 uses the linear sieve functions F_1, f_1 (dhr_sieve.DHR(1)), kappa = 2 the DHR functions F_2, f_2.
Floating point.  Usage: python dimension_benchmark.py [theta ...]
"""
import os
import sys
import numpy as np
from math import ceil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from dhr_sieve import DHR   # noqa: E402

thetas = [float(eval(t)) for t in sys.argv[1:]] or [0.5, 2 / 3, 0.8, 0.95, 1.0]


def LR(D, kappa, theta, a, b, nu=600):
    L = float(D.fs(theta / a))
    t = np.linspace(a, b, nu + 1)
    R = kappa * float(np.trapz((1 - t / b) * D.Fs((theta - t) / a) / t, t))
    return L, R


def best(D, kappa, theta):
    rows = []
    for a in np.arange(0.01, theta / D.beta - 1e-9, 0.001):
        for b in np.arange(0.10, theta - 0.0049, 0.0025):
            if b <= a:
                continue
            L, R = LR(D, kappa, theta, a, b)
            if L <= 0:
                continue
            rows.append((R / L + 2 / b, R / L + 1 / b, a, b, L, R))
    jt = min(rows)
    sd = min(rows, key=lambda r: r[1])
    return jt, sd


def pair(jt, sd):
    """(k, k') from the thresholds: Omega <= k needs threshold <= k+1 (strictly below in the limit)."""
    return ceil(jt - 1 - 1e-9), ceil(sd - 1 - 1e-9)


if __name__ == "__main__":
    out = []
    D1 = DHR(1, guess=(2.0, 2.0), verbose=False)
    D2 = DHR(2, verbose=False)
    out.append(f"linear sieve: alpha_1 = beta_1 = {D1.beta:.4f};  dimension 2: alpha_2 = {D2.alpha:.4f}, beta_2 = {D2.beta:.4f}")
    head2 = "kappa=2 (weights r r): joint, side -> (k,k'), at (a,b)"
    head1 = "kappa=1 (indicator weights): joint, side -> (k,k'), at (a,b)"
    out.append("%6s | %-72s | %-72s" % ("theta", head2, head1))
    for th in thetas:
        jt2, sd2 = best(D2, 2, th)
        jt1, sd1 = best(D1, 1, th)
        k2 = pair(jt2[0], sd2[1])
        k1 = pair(jt1[0], sd1[1])
        out.append(f"{th:>6.3f} | joint {jt2[0]:6.3f} (a={jt2[2]:.3f}, b={jt2[3]:.3f}), side {sd2[1]:6.3f} (a={sd2[2]:.3f}, b={sd2[3]:.3f}) "
                   f"-> ({k2[0]},{k2[1]}) | joint {jt1[0]:6.3f} (a={jt1[2]:.3f}, b={jt1[3]:.3f}), side {sd1[1]:6.3f} "
                   f"(a={sd1[2]:.3f}, b={sd1[3]:.3f}) -> ({k1[0]},{k1[1]})")
    out.append("\n(k, k'): Omega(n(N-n)) <= k and max(Omega(n), Omega(N-n)) <= k', from min(1/lam + 2/b) <= k + 1 and "
               "min(1/lam + 1/b) <= k' + 1 with L - lam R > 0; the two minima are attained at different (a, b), so a "
               "simultaneous pair may need one more unit on one side (Part II, Table 3).  The kappa = 1 column assumes "
               "a level of distribution theta for the indicator weights b(n) b(N-n), which no theta series provides; "
               "it is the benchmark, not a theorem.")
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
    with open(os.path.join(HERE, '..', 'results', 'dimension_benchmark.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")
