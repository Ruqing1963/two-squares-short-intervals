r"""
Research 1 (Part III notes, Section 1): the symmetric remainder of the two-dimensional sieve.

For N = 2 mod 4, A = {n <= N : (n, N) = 1}, w(n) = r(n) r(N - n), and coprime odd square-free d1, d2 with
(d1 d2, N) = 1,
    S(d1, d2) = sum_{n in A, d1 | n, d2 | N - n} w(n) = X_N h(d1) h(d2) + R(d1, d2; N)          (Part I, Section 3),
    R_d       = sum_{d1 d2 = d} R(d1, d2; N)                                                   (Part II, Section 7.2),
R_d being the remainder of the sequence {n in A : d | n(N-n)} that the two-dimensional sieve actually uses.  Section 7.2
bounds |R_d| by sum_{d1 d2 = d} |R(d1, d2; N)|.  By the Atkin-Lehner identity
    sum_{u | d} G_{t u, d/u} = prod_{p | d} (1 + W_p) G_{t, d}        (notes, Lemma 1.1; chi(u) = 1 for the u that occur),
the symmetric sum sees only the joint (+1)-eigenspace of the W_p, p | d, so that every Hecke component with
eta_p(f) = -1 for some p | (d, M_1) cancels exactly.  Since R(d1, d2) = R(d2, d1) (n <-> N - n), nothing cancels for
d prime; for omega(d) >= 2 the cancellation is between the 2^{omega(d) - 1} unordered splittings.

This script computes every R(d1, d2; N) exactly (direct summation; X_N from Part I, (17)-(18)) for all admissible pairs
with d1 d2 <= N^0.75 and compares, for D = N^theta,
    A_abs = sum_{d1 d2 <= D} |R(d1, d2)|,     A_sym = sum_{d <= D} |R_d|,
    the same with the sieve weight 4^omega(d),  Q = root sums of squares,
    A_rnd = sum_d |sum_{splittings} (random sign) 2 R(u, d/u)|  (the random-sign baseline, averaged over draws),
in total and separately for omega(d) = 1, 2, 3; and, across all the N in the list, the empirical share
    s_{++}(d) = sum_N (R(d,1) + R(p1,p2))^2 / sum_N 2 ((R(d,1))^2 + (R(p1,p2))^2)       (d = p1 p2),
which the notes predict to be ||P^+ G_{1,d}||^2 / ||G_{1,d}||^2 (the share of the norm in the joint (+1)-eigenspace;
compare symmetric_norm_share.py).  Floating point.
Usage: python symmetric_remainder.py [N ...]      (default: 24 values of N = 2 mod 4 near 2 10^6)
"""
import os
import sys
import time
import numpy as np
from math import gcd, pi, log, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from sup_error_scan import r_table, factor   # noqa: E402

Ns = [int(a) for a in sys.argv[1:]] or [2000002, 2000006, 2000018, 3000002] + [2000030 + 4 * j for j in range(20)]
THETAS = (0.5, 0.6, 0.7, 0.75)
THETA_MAX = 0.75
rng = np.random.default_rng(7)
N_RND = 40


def h(p):
    return (2 * p - 1) / (p * (p + 1)) if p % 4 == 1 else 1.0 / (p * (p - 1))


def X_N(N):
    """The Eisenstein part of S(1,1) = sum_{n in A} w(n): Part I, (17) with Lemma 3.1(ii)."""
    M = N // 2
    sig = 1
    for p, k in factor(M).items():
        sig *= (p ** (k + 1) - 1) // (p - 1)
    X = 16.0 * sig
    for p, a in factor(N).items():
        if p == 2:
            continue
        bstar = (1 - 1 / p) ** 3 if p % 4 == 1 else (1 - 1 / p) * (1 + 1 / p) ** 2
        sigm1 = (1 - p ** -(a + 1.0)) / (1 - 1.0 / p)
        g = 1 - bstar / ((1 - p ** -2.0) * sigm1)
        X *= 1 - g
    return X


def mobius_sqfree_odd(D):
    mu = np.ones(D + 1, dtype=np.int64)
    sqf = np.ones(D + 1, dtype=bool)
    om = np.zeros(D + 1, dtype=np.int64)
    isp = np.ones(D + 1, dtype=bool)
    isp[:2] = False
    for p in range(2, D + 1):
        if not isp[p]:
            continue
        isp[p * p::p] = False
        mu[p::p] *= -1
        om[p::p] += 1
        sqf[p * p::p * p] = False
    mu[~sqf] = 0
    sqf[::2] = False
    sqf[0] = False
    return mu, sqf, om


def remainders(N):
    """All R(d1, d2; N) for coprime odd square-free d1, d2 with (d1 d2, N) = 1, d1 d2 <= N^THETA_MAX."""
    Dmax = int(N ** THETA_MAX) + 1
    r = r_table(N).astype(np.float64)
    w = r * r[::-1]                      # w[n] = r(n) r(N - n)
    cop = np.ones(N + 1, dtype=bool)
    for p in factor(N):
        cop[::p] = False
    w[~cop] = 0.0
    X = X_N(N)
    S11 = w.sum()
    mu, sqf, om = mobius_sqfree_odd(Dmax)
    rows = []
    for d1 in range(1, Dmax + 1, 2):
        if not sqf[d1] or gcd(d1, N) != 1:
            continue
        h1 = h_mult(d1)
        for d2 in range(1, Dmax // d1 + 1, 2):
            if not sqf[d2] or gcd(d1, d2) != 1 or gcd(d2, N) != 1:
                continue
            d = d1 * d2
            if d2 > 1:
                k0 = (N * pow(d1, -1, d2)) % d2
                n0 = d1 * k0
            else:
                n0 = 0
            if n0 == 0:
                n0 = d
            S = w[n0::d].sum()
            R = S - X * h1 * h_mult(d2)
            rows.append((d, d1, d2, R))
    rows.sort()
    return rows, X, S11, om


_hcache = {}


def h_mult(d):
    if d not in _hcache:
        v = 1.0
        for p in factor(d):
            v *= h(p)
        _hcache[d] = v
    return _hcache[d]


def analyse(rows, om, N, out):
    d = np.array([x[0] for x in rows], dtype=np.int64)
    d1 = np.array([x[1] for x in rows], dtype=np.int64)
    d2 = np.array([x[2] for x in rows], dtype=np.int64)
    R = np.array([x[3] for x in rows])
    # symmetry check R(d1,d2) = R(d2,d1)
    key = {(a, b): v for a, b, v in zip(d1, d2, R)}
    asym = max(abs(key[(a, b)] - key[(b, a)]) for a, b in zip(d1, d2))
    # group by d
    uniq, start = np.unique(d, return_index=True)
    Rd = np.add.reduceat(R, start)
    omd = om[uniq]
    # random-sign baseline over unordered splittings: terms 2 R(u, d/u) with u < d/u
    lower = d1 < d2
    dl, Rl = d[lower], 2 * R[lower]
    ul, sl = np.unique(dl, return_index=True)
    assert np.array_equal(ul, uniq[uniq > 1])
    rnd = np.zeros((N_RND, len(uniq)))
    for t in range(N_RND):
        sg = rng.choice([-1.0, 1.0], size=len(Rl))
        rnd[t, 1:] = np.abs(np.add.reduceat(sg * Rl, sl))
    rnd[:, 0] = abs(Rd[0])
    out.append(f"N = {N} = {'*'.join(str(p) + ('^%d' % v if v > 1 else '') for p, v in sorted(factor(N).items()))}: "
               f"{len(rows)} pairs, {len(uniq)} moduli d <= N^{THETA_MAX};  max |R(d1,d2) - R(d2,d1)| = {asym:.1e}")
    out.append(f"   {'theta':>5} {'omega':>5} {'#d':>6} {'A_abs/N':>9} {'A_sym/N':>9} {'A_rnd/N':>9} {'sym/abs':>8} {'sym/rnd':>8} "
               f"{'W4_sym/W4_abs':>13} {'Q_sym/Q_abs':>11}")
    table = {}
    for th in THETAS:
        D = N ** th
        for omega in (0, 1, 2, 3):
            if omega == 0:
                sel = uniq <= D
                selp = d <= D
                lab = 'all'
            else:
                sel = (uniq <= D) & (omd == omega)
                selp = (d <= D) & (om[d] == omega)
                lab = str(omega)
            if sel.sum() == 0:
                continue
            A_abs = np.abs(R[selp]).sum()
            A_sym = np.abs(Rd[sel]).sum()
            A_rnd = rnd[:, sel].sum(axis=1).mean()
            W_abs = (4.0 ** om[d[selp]] * np.abs(R[selp])).sum()
            W_sym = (4.0 ** omd[sel] * np.abs(Rd[sel])).sum()
            Q_abs = sqrt((R[selp] ** 2).sum())
            Q_sym = sqrt((Rd[sel] ** 2).sum())
            out.append(f"   {th:>5.2f} {lab:>5} {sel.sum():>6} {A_abs / N:>9.4f} {A_sym / N:>9.4f} {A_rnd / N:>9.4f} "
                       f"{A_sym / A_abs:>8.3f} {A_sym / A_rnd:>8.3f} {W_sym / W_abs:>13.3f} {Q_sym / Q_abs:>11.3f}")
            table[(th, omega)] = (A_sym / A_abs, A_sym / A_rnd, W_sym / W_abs)
    return key, table


if __name__ == "__main__":
    t0 = time.time()
    out = []
    keys = []
    tables = []
    for N in Ns:
        rows, X, S11, om = remainders(N)
        out.append(f"\nN = {N}: X_N = {X:.1f}, S(1,1) = {S11:.0f}, R(1,1)/sqrt(N) = {(S11 - X) / sqrt(N):+.3f}  "
                   f"[{time.time() - t0:.0f}s]")
        key, table = analyse(rows, om, N, out)
        keys.append(key)
        tables.append(table)
        print("\n".join(out[-10:]), flush=True)
    # averages of the ratios over the N
    out.append("\nAverages over the %d values of N of the ratios A_sym/A_abs, A_sym/A_rnd, W4_sym/W4_abs:" % len(Ns))
    for th in THETAS:
        for omega in (0, 1, 2, 3):
            vals = [t[(th, omega)] for t in tables if (th, omega) in t]
            if not vals:
                continue
            v = np.array(vals)
            out.append(f"   theta = {th:.2f}, omega = {'all' if omega == 0 else omega}: "
                       f"sym/abs = {v[:, 0].mean():.3f} +- {v[:, 0].std():.3f}, sym/rnd = {v[:, 1].mean():.3f} +- {v[:, 1].std():.3f}, "
                       f"W4 sym/abs = {v[:, 2].mean():.3f}")
    # empirical s_{++}(d) for d = p1 p2 across the N: (R(d,1) + R(p1,p2))^2 / (2 (R(d,1)^2 + R(p1,p2)^2))
    out.append("\nEmpirical joint-(+1) share s_{++}(d) = <(R(d,1)+R(p1,p2))^2> / <2(R(d,1)^2 + R(p1,p2)^2)> over the N, "
               "for d = p1 p2 <= 2000 (both primes = 1 mod 4 first, then mixed):")
    ds = sorted({dd for k in keys for (a, b) in [k] for dd in [a * b]} if False else
                {a * b for k in keys for (a, b) in k if a < b and len(factor(a * b)) == 2 and a * b <= 2000})
    rows_s = []
    for dd in ds:
        p1, p2 = sorted(factor(dd))
        num = den = 0.0
        cnt = 0
        for k in keys:
            if (1, dd) in k and (p1, p2) in k:
                x, y = k[(1, dd)], k[(p1, p2)]
                num += (x + y) ** 2
                den += 2 * (x * x + y * y)
                cnt += 1
        if cnt:
            rows_s.append((dd, p1, p2, num / den, cnt))
    split = [r for r in rows_s if r[1] % 4 == 1 and r[2] % 4 == 1]
    mixed = [r for r in rows_s if not (r[1] % 4 == 1 and r[2] % 4 == 1)]
    for lab, grp in (("both split", split), ("at least one inert", mixed)):
        out.append(f"   {lab}: {len(grp)} moduli, mean s_++ = {np.mean([g[3] for g in grp]):.3f} "
                   f"(random signs would give 0.5)")
        out.append("      " + "  ".join(f"{g[0]}={g[1]}*{g[2]}:{g[3]:.2f}" for g in grp[:30]))
    out.append(f"\ntotal time {time.time() - t0:.0f}s")
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
    with open(os.path.join(HERE, '..', 'results', 'symmetric_remainder.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")
