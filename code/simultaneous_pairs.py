r"""
Which pairs (k, k') are attainable SIMULTANEOUSLY (one set of parameters (a, b, lam)) by the weighted sieve of
dimension kappa at the level theta?  A pair is attainable if there are a < b < theta, a < theta/beta_kappa, with
    L - lam R > 0,   L = f_kappa(theta/a),   R = kappa int_a^b (1 - t/b) F_kappa((theta - t)/a) dt/t,
    1/lam = min(k + 1 - 2/b, k' + 1 - 1/b) > 0.
The separate minima of 1/lam + 2/b and 1/lam + 1/b (dimension_benchmark.py) are attained at different (a, b) and
need not be compatible: e.g. (5,3) needs 1/b <= 2, i.e. b >= 1/2, impossible for theta <= 1/2.  The script prints,
for kappa = 1, 2 and several theta, the minimal attainable pairs (Pareto set) with their best margins.
Usage: python simultaneous_pairs.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from dhr_sieve import DHR   # noqa: E402


def margin(D, kappa, theta, k, kp):
    best = (-9.0, None, None)
    for a in np.arange(0.01, theta / D.beta - 1e-9, 0.002):
        L = float(D.fs(theta / a))
        if L <= 0:
            continue
        for b in np.arange(0.10, theta - 0.0049, 0.0025):
            inv = min(k + 1 - 2 / b, kp + 1 - 1 / b)
            if inv <= 0:
                continue
            t = np.linspace(a, b, 400)
            R = kappa * float(np.trapz((1 - t / b) * D.Fs((theta - t) / a) / t, t))
            m = L - R / inv
            if m > best[0]:
                best = (m, a, b)
    return best


if __name__ == "__main__":
    out = []
    Ds = {1: DHR(1, guess=(2.0, 2.0), verbose=False), 2: DHR(2, verbose=False)}
    for kappa in (2, 1):
        for theta in (0.5, 2 / 3, 0.8, 0.95, 1.0):
            att = {}
            for k in range(2, 9):
                for kp in range(1, 6):
                    if kp > k or 2 * kp < k:
                        continue
                    m, a, b = margin(Ds[kappa], kappa, theta, k, kp)
                    if m > 0:
                        att[(k, kp)] = (m, a, b)
            pareto = [p for p in att if not any((q[0] <= p[0] and q[1] <= p[1] and q != p) for q in att)]
            txt = "; ".join(f"({k},{kp}) margin {att[(k, kp)][0]:+.3f} at (a,b) = ({att[(k, kp)][1]:.3f}, {att[(k, kp)][2]:.3f})"
                            for (k, kp) in sorted(pareto))
            out.append(f"kappa = {kappa}, theta = {theta:.4f}: minimal attainable pairs: {txt}")
            print(out[-1], flush=True)
    with open(os.path.join(HERE, '..', 'results', 'simultaneous_pairs.txt'), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(out) + "\n")
