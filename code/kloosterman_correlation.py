r"""
Exploration A (Part III notes, Section 4): the sizes in the Cauchy-Schwarz / Poisson attack on the fixed first argument.

In the trilinear form (5) of the notes the Kloosterman sums S(16N, m|h|^2; m'k) are summed over a box |h| <= H with
H = m'k sqrt(m/N) (so H^2 = m (m'k)^2 / N <= m m' at the top of the k-range), while the modulus in the Poisson step for
the off-diagonal terms (m'_1, m'_2) is q = [m'_1 k, m'_2 k] ~ (m'k)^2 / gcd.  The script prints, for N = 2000002, m = 5
and k in the Kloosterman regime k >= sqrt(N/m)/m', the ratio H^2 / q (always << 1: the box is far shorter than the
modulus, so Poisson in h lengthens the sum), and measures the correlation matrix
    C(m'_1, m'_2) = sum_h W(h/H) S(16N, m|h|^2; m'_1 k) S(16N, m|h|^2; m'_2 k)
over m' in a set of primes = 1 mod 4: the diagonal C(m',m') against the Weil size H^2 m'k, the off-diagonal entries
against sqrt(C(m'_1,m'_1) C(m'_2,m'_2)) / sqrt(#h) (the random-sign size), and the quadratic form with the
signs mu(m') and random signs against the diagonal (square-root cancellation over m' means |sum_{m'} +- sum_h W S|^2
is of the size of the diagonal sum).  Floating point.
"""
import sys
import numpy as np
from math import gcd, sqrt, pi, exp

N = 2000002
m = 5
primes = [41, 53, 61, 73, 89, 97, 109, 113, 137, 149, 157, 173]      # m' = 1 mod 4, coprime to m and N (101 | N excluded)
ks = [43, 83, 127]                                                  # odd, as the true moduli m'k are (k | beta, beta odd)
rng = np.random.default_rng(3)


def kloosterman_table(a, c, nvals):
    """S(a, n; c) for all n in nvals (direct summation over t mod c)."""
    t = np.array([x for x in range(1, c) if gcd(x, c) == 1])
    tbar = np.array([pow(int(x), -1, c) for x in t])
    base = np.exp(2j * pi * (a % c) * t / c)
    out = {}
    for n in set(int(v) for v in nvals):
        out[n] = np.sum(base * np.exp(2j * pi * (n % c) * tbar / c)).real
    return out


for k in ks:
    print(f"k = {k}:")
    rows = {}
    hs = {}
    for mp in primes:
        c = mp * k
        if c * c < N / m:                       # below the Kloosterman regime: no h != 0 term survives
            continue
        H = c * sqrt(m / N)
        R = int(2.5 * H) + 1
        h1, h2 = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))
        j = (h1 ** 2 + h2 ** 2).ravel()
        par = ((h1 - h2) % 2 == 0).ravel()
        sel = (j > 0) & par
        j = j[sel]
        W = np.exp(-pi * j / H ** 2)
        tab = kloosterman_table(16 * N, c, m * j)
        S = np.array([tab[int(m * v)] for v in j])
        rows[mp] = (W, S, H, c)
    mps = sorted(rows)
    if len(mps) < 3:
        print("   too few moduli in the Kloosterman regime")
        continue
    # all rows must use the same h-set for the correlation matrix: use the smallest box (first m') as the common support
    # (the weights W already localise to |h| <= H_{m'}; we take the intersection of the lattices by recomputing on the
    # largest box and letting W do the cut)
    Rmax = max(int(2.5 * rows[mp][2]) + 1 for mp in mps)
    h1, h2 = np.meshgrid(np.arange(-Rmax, Rmax + 1), np.arange(-Rmax, Rmax + 1))
    j = (h1 ** 2 + h2 ** 2).ravel()
    sel = (j > 0) & (((h1 - h2) % 2 == 0).ravel())
    j = j[sel]
    vec = {}
    for mp in mps:
        c = mp * k
        H = c * sqrt(m / N)
        W = np.exp(-pi * j / H ** 2)
        tab = kloosterman_table(16 * N, c, m * j)
        S = np.array([tab[int(m * v)] for v in j])
        vec[mp] = np.sqrt(W) * S          # so that C = vec . vec
        q_typ = c * c
        print(f"   m' = {mp:4d}: c = m'k = {c:6d}, H = {H:6.2f}, #h (eff.) = {np.sum(W):7.1f}, H^2/c^2 = {H * H / q_typ:.1e}, "
              f"diag C/(H^2 c) = {np.sum(W * S * S) / (np.sum(W) * c):.3f}")
    M = np.array([[np.dot(vec[a], vec[b]) for b in mps] for a in mps])
    d = np.sqrt(np.diag(M))
    corr = M / np.outer(d, d)
    off = corr[~np.eye(len(mps), dtype=bool)]
    neff = np.mean([np.sum(np.exp(-pi * j / (mp * k * sqrt(m / N)) ** 2)) for mp in mps])
    print(f"   off-diagonal correlations: rms {np.sqrt(np.mean(off ** 2)):.3f}, max |.| {np.abs(off).max():.3f}; "
          f"random-sign size 1/sqrt(#h) = {1 / sqrt(neff):.3f}")
    mu = np.array([-1.0 for _ in mps])           # mu(m') = -1 for primes
    qf_mu = mu @ M @ mu / np.trace(M)
    qf_rnd = np.mean([(s @ M @ s) / np.trace(M) for s in rng.choice([-1.0, 1.0], size=(200, len(mps)))])
    qf_one = np.ones(len(mps)) @ M @ np.ones(len(mps)) / np.trace(M)
    print(f"   quadratic form / trace: signs mu {qf_mu:.2f}, all +1 {qf_one:.2f}, random signs (mean of 200) {qf_rnd:.2f}  "
          f"(1 = square-root cancellation over m', {len(mps)} = none)")
    sys.stdout.flush()
