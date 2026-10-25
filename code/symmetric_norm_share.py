r"""
Research 1 (Part III notes, Section 1): the joint Atkin-Lehner eigenspaces of G_{1,d}.

For odd square-free d with prime factors p_1, ..., p_k, Lemma 1.1 of the notes gives G_{u, d/u} = chi(u) G_{1,d} | W_u, so
the projection of G_{1,d} onto the joint eigenspace with signs eps = (eps_p)_{p | d} is
    P_eps G_{1,d} = 2^{-k} sum_{u | d} eps(u) chi(u) G_{u, d/u},      eps(u) = prod_{p | u} eps_p,
and W_d = prod W_p acts on G_{1,d} as chi(d), so only the patterns with prod eps_p = chi(d) occur.  The script computes
the coefficients a_{G_{u,d/u}}(n), n <= X, exactly (local_densities.siegel_all) for every u | d and the shares
    s_eps(d) = ||P_eps G_{1,d}||^2 / ||G_{1,d}||^2       (Rankin-Selberg: sum_{n <= X} |a(n)|^2 / n, same X),
checks that the patterns with the wrong product carry no norm, and prints the share of the all-(+1) pattern, which is
what the symmetric remainder R_d of the sieve sees (symmetric_remainder.py measures it as s_{++}).
Usage: python symmetric_norm_share.py [d ...]
"""
import os
import sys
import time
import itertools
import numpy as np
from math import pi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from sup_error_scan import r_table, sigma_over_n_odd, factor, chi4   # noqa: E402
from local_densities import siegel_all                                 # noqa: E402

ds = [int(a) for a in sys.argv[1:]] or ([65, 85, 145, 185, 205, 221, 265, 305, 365, 377, 445, 481, 493, 533, 629, 689, 697, 901]
                                       + [15, 21, 33, 35, 39, 51, 55, 77, 91, 95, 143, 187]
                                       + [1105, 1885, 2465, 1365, 2145, 3315])
X = 2 ** 18


def divisors_sqfree(d):
    ps = sorted(factor(d))
    out = []
    for bits in itertools.product((0, 1), repeat=len(ps)):
        u = 1
        for b, p in zip(bits, ps):
            if b:
                u *= p
        out.append((bits, u))
    return ps, out


if __name__ == "__main__":
    t0 = time.time()
    r = r_table(X)
    fs, v2 = sigma_over_n_odd(X)
    nn = np.arange(X + 1, dtype=float)
    lines = []
    for d in ds:
        ps, divs = divisors_sqfree(d)
        k = len(ps)
        coef = {}
        for bits, u in divs:
            T, M = siegel_all(u, d // u, X, r, fs, v2)
            coef[bits] = (T - M)[: X + 1]
        base = coef[tuple([0] * k)]
        norm = np.sum(base[1:] ** 2 / nn[1:])
        # symmetry G_{u,d/u} = G_{d/u,u}
        sym = max(np.abs(coef[bits] - coef[tuple(1 - b for b in bits)]).max() for bits, u in divs)
        shares = {}
        comps = {}
        for eps in itertools.product((1, -1), repeat=k):
            acc = np.zeros(X + 1)
            for bits, u in divs:
                sgn = chi4(u) if u > 1 else 1
                for b, e in zip(bits, eps):
                    if b:
                        sgn *= e
                acc += sgn * coef[bits]
            acc /= 2 ** k
            comps[eps] = acc
            shares[eps] = np.sum(acc[1:] ** 2 / nn[1:]) / norm
        chid = chi4(d)
        wrong = max(v for e, v in shares.items() if np.prod(e) != chid)
        allowed = [e for e in shares if np.prod(e) == chid]
        # the same shares restricted to the coefficients the sieve sees: n = 2 mod 4 (N itself; the newforms of
        # level 4d have a(2) = 0 and contribute nothing there), n = 1 mod 4 (N/2), and the sieve combination
        # a(2n') - a(n') for n' = 1 mod 4 coprime to d
        n1 = np.arange(1, X // 2 + 1, 4)
        n1 = n1[np.gcd(n1, d) == 1]
        cls = {}
        for lab, idx in (("n=2(4)", 2 * n1), ("n=1(4)", n1)):
            tot = sum(np.sum(comps[e][idx] ** 2 / idx) for e in allowed)
            cls[lab] = {e: np.sum(comps[e][idx] ** 2 / idx) / tot for e in allowed}
        tot = sum(np.sum((comps[e][2 * n1] - comps[e][n1]) ** 2 / n1) for e in allowed)
        if tot > 0:
            cls["a(2n')-a(n')"] = {e: np.sum((comps[e][2 * n1] - comps[e][n1]) ** 2 / n1) / tot for e in allowed}
        else:
            # chi(d) = -1: a(2n') = a(n') for every odd n' (T^odd(m,m';2n') = 0 when chi(mm') = -1, Part I, (16));
            # such pairs do not occur in the sieve decomposition
            cls["a(2n')-a(n')"] = None

        def fmt(dct):
            return ", ".join("(" + "".join('+' if s > 0 else '-' for s in e) + f") {v:.3f}" for e, v in dct.items())
        lines.append(f"d = {d} = {'*'.join(map(str, ps))} (chi(d) = {chid:+d}): norm shares {fmt({e: shares[e] for e in allowed})};  "
                     f"forbidden patterns carry <= {wrong:.1e};  max |a_(u,d/u) - a_(d/u,u)| = {sym:.1e}   "
                     f"[{time.time() - t0:.0f}s]")
        for lab in ("n=2(4)", "n=1(4)", "a(2n')-a(n')"):
            if cls[lab] is None:
                lines.append(f"      shares over {lab:>13}: identically zero (chi(d) = -1: a(2n') = a(n') for odd n')")
            else:
                lines.append(f"      shares over {lab:>13}: {fmt(cls[lab])}")
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
    with open(os.path.join(HERE, '..', 'results', 'symmetric_norm_share.txt'), 'w', encoding='utf-8') as fh:
        fh.write(f"Rankin-Selberg shares over n <= {X}\n" + text + "\n")
