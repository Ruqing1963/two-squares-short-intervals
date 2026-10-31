r"""
Pair weights in the two-dimensional weighted sieve at the level theta (Part III, Section 6.4): does a non-separable
weight
    W(n) = 1 - sum_{p | n(N-n)} g_1(t_p) - sum_{p < q, pq | n(N-n)} g_2(t_p, t_q),      t_p = log p / log N,
(lambda, mu absorbed into g_1, g_2 >= 0; all prime factors of n(N-n) at least z = N^a) close the 0.030 gap to (7,4)?

Lower bound.  The upper-bound sieve of dimension two applied to S(A_p, z) at the level N^{theta - t_p} and to
S(A_{pq}, z) at the level N^{theta - t_p - t_q}, with the densities 2h(p), 2h(p)2h(q) and sum_p 2h(p) phi(t_p) ->
int phi(t) 2 dt/t, gives
    W >= X_N V(z) ( f_2(theta/a) - R_1(g_1) - R_2(g_2) - o(1) ),
    R_1(g_1) = 2 int g_1(t) F_2((theta - t)/a) dt/t,
    R_2(g_2) = 4 int int_{u < v} g_2(u, v) F_2((theta - u - v)/a) du dv/(uv)      (u = v: the half square).
Positivity.  With
    G_k(S) = inf { sum_i g_1(t_i) + sum_{i<j} g_2(t_i, t_j) : t_i >= a, t_1 + ... + t_k <= S },
non-decreasing in k and non-increasing in S (g_1, g_2 >= 0), the weight is <= 0 on every square-free n with
Omega(n) > k' or Omega(N-n) > k' or Omega(n(N-n)) > k as soon as
    G_{k'+1}(1) >= 1   and   G_{k1}(1) + G_{k2}(1) >= 1 for k1 + k2 = k + 1, k1, k2 <= k'
(in the joint condition the pairs with one prime in n and one in N-n are dropped; they are >= 0, so the condition is
sufficient, and it is the exact one when the joint constraint does not bind).  The best margin is
    f_2(theta/a) - min { R_1(g_1) + R_2(g_2) : the constraints above, g_1, g_2 >= 0 }.

Discretisation (a and 1 multiples of the step delta, nodes t_l = a + l delta).  g_1 is continuous and piecewise
linear with nodes t_l (as in optimal_weights.py), g_2 is constant on the products of cells [t_l, t_{l+1}) x
[t_m, t_{m+1}) and non-increasing in each argument.  Then the infimum of the penalty over real configurations
equals its minimum over node configurations (t_i in {t_l}, sum <= 1): on each cell product the g_2 part is
constant and the g_1 part is linear, so the infimum is at a vertex of (cell product) cap {sum t_i <= 1}, whose
coordinates are nodes (1 - ka is a multiple of delta); a coordinate at the right end of its cell is the node of the
next cell, where g_2 is not larger.  So the constraints carry no grid artefact, and with g_2 = 0 the programme is the
one of optimal_weights.py (baseline, margin -0.030 for (7,4)).  Because the pair term is not additive over the
primes, dynamic programming no longer applies: all k-configurations (sorted nodes, sum <= 1) are enumerated
(about 10^6 for k = 4 and 5 10^6 for k = 5 at delta = 0.005) and the LP in the node values of g_1 and the cell
values of g_2 is solved by cutting planes, the most violated configurations being added at each round.  At the
optimum the reduced cost of each pair cell (its cost minus the dual value of the constraints it tightens) says how
much cheaper the pair term would have to be to enter the optimal weight.  The flag F2=1 replaces every F_2 by 1
(no sieve cost: the pure moment inequality for the Poisson model of the prime factors), which separates the
combinatorial from the sieve effect; mono=False drops the monotonicity of g_2 (then the node check is only
necessary, not sufficient, for real configurations).  Floating point.
Usage: python optimal_weights_2d.py [theta] [delta]
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two-squares-mean-square', 'code'))
from dhr_sieve import DHR          # noqa: E402

theta = float(eval(sys.argv[1])) if len(sys.argv) > 1 else 0.5
DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
D2 = DHR(2, verbose=False)
RES = os.path.join(HERE, '..', 'results')


def enumerate_configs(k, B):
    """All non-decreasing k-tuples (l_1 <= ... <= l_k) of non-negative integers with sum <= B, as an (M, k) int32 array."""
    if B < 0:
        return np.zeros((0, k), dtype=np.int32)
    if k == 1:
        return np.arange(B + 1, dtype=np.int32)[:, None]
    blocks = []

    def last_two(prefix, lmin, rem):
        xs = np.arange(lmin, rem // 2 + 1)
        if len(xs) == 0:
            return
        counts = rem - 2 * xs + 1
        x = np.repeat(xs, counts)
        start = np.cumsum(counts) - counts
        y = x + (np.arange(counts.sum()) - np.repeat(start, counts))
        blk = np.empty((len(x), k), dtype=np.int32)
        blk[:, :k - 2] = prefix
        blk[:, k - 2] = x
        blk[:, k - 1] = y
        blocks.append(blk)

    def rec(prefix, lmin, rem, left):
        if left == 2:
            last_two(prefix, lmin, rem)
            return
        for l in range(lmin, rem // left + 1):
            rec(prefix + [l], l, rem - l, left - 1)

    rec([], 0, B, k)
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, k), dtype=np.int32)


class PairSieve:
    def __init__(self, a, pairs=True, s_min=1.0, b1=None, flat=False, mono=True, discount=0.0):
        """Variables: g_1 at the nodes t_l < b1 (default b1 = theta - 2 delta; g_1 = 0 at the nodes >= b1), g_2 on the
        cell pairs l <= m whose upper corner satisfies (theta - u - v)/a >= s_min.  flat: F_2 := 1 in the costs;
        discount: the pair costs are multiplied by 1 - discount (a hypothetical cheaper pair term, to measure how far
        the pair cells are from entering the optimal weight)."""
        ia = a / DELTA
        assert abs(ia - round(ia)) < 1e-9 and abs(1 / DELTA - round(1 / DELTA)) < 1e-9, "a and 1 must be multiples of delta"
        self.a, self.mono = a, mono
        self.b1 = theta - 2 * DELTA if b1 is None else b1
        self.ncell = int(round((1.0 - a) / DELTA)) + 1
        self.tl = a + DELTA * np.arange(self.ncell)                        # nodes = left endpoints of the cells
        self.n1 = int(np.sum(self.tl < self.b1 - 1e-12))
        # cost of the hat function at node l: int hat_l(t) F_2((theta - t)/a) 2 dt/t (midpoint rule, 40 points per cell)
        sub = (np.arange(40) + 0.5) / 40
        self.cost1 = np.zeros(self.n1)
        for l in range(self.n1):
            for cell, ramp in ((l - 1, sub), (l, 1 - sub)):             # rising part on [t_{l-1}, t_l], falling on [t_l, t_{l+1}]
                if cell < 0:
                    continue
                t = self.tl[cell] + DELTA * sub
                F = np.ones_like(t) if flat else D2.Fs((theta - t) / a)
                self.cost1[l] += float(np.sum(ramp * F * 2 * DELTA / 40 / t))
        mid = self.tl + DELTA / 2
        mass = 2 * np.log((self.tl + DELTA) / self.tl)                     # int_cell 2 dt/t
        self.pidx = np.full((self.ncell, self.ncell), -1, dtype=np.int32)
        cost2, pl, pm = [], [], []
        if pairs:
            for l in range(self.n1):
                for m in range(l, self.n1):
                    if self.tl[l] + self.tl[m] + 2 * DELTA > theta - s_min * a + 1e-12:
                        break
                    s = (theta - mid[l] - mid[m]) / a
                    F = 1.0 if flat else float(D2.Fs(s))
                    cost2.append(F * mass[l] * mass[m] * (0.5 if l == m else 1.0))
                    self.pidx[l, m] = self.pidx[m, l] = len(pl)
                    pl.append(l)
                    pm.append(m)
        self.n2 = len(pl)
        self.pidx[self.pidx < 0] = self.n2                                 # the zero slot
        self.cost2 = np.array(cost2, dtype=float) * (1.0 - discount)
        self.pl, self.pm = np.array(pl, dtype=np.int32), np.array(pm, dtype=np.int32)
        self.cost = np.concatenate([self.cost1, self.cost2])
        self.n = self.n1 + self.n2
        self.L, self.PI = {}, {}
        self.res = None

    # ---------------------------------------------------------------- configurations
    def configs(self, k):
        """All node configurations of k primes with sum <= 1, and their pair-variable indices."""
        if k not in self.L:
            B = int(round((1.0 - k * self.a) / DELTA))
            L = enumerate_configs(k, B)
            self.L[k] = L
            if self.n2 and k <= 5:
                cols = [self.pidx[L[:, i], L[:, j]] for i in range(k) for j in range(i + 1, k)]
                self.PI[k] = np.stack(cols, axis=1)
        return self.L[k]

    def penalty(self, x, k):
        L = self.configs(k)
        g1 = np.zeros(self.ncell)
        g1[:self.n1] = x[:self.n1]
        P = g1[L].sum(axis=1)
        if self.n2:
            g2 = np.append(x[self.n1:], 0.0)
            if k in self.PI:
                P += g2[self.PI[k]].sum(axis=1)
            else:
                for i in range(k):
                    for j in range(i + 1, k):
                        P += g2[self.pidx[L[:, i], L[:, j]]]
        return P

    def Gk(self, x, k):
        P = self.penalty(x, k)
        i = int(np.argmin(P))
        return float(P[i]), [float(self.tl[l]) for l in self.configs(k)[i]]

    def row(self, conf):
        cols = {}
        for l in conf:
            if l < self.n1:
                cols[int(l)] = cols.get(int(l), 0) + 1
        for i in range(len(conf)):
            for j in range(i + 1, len(conf)):
                p = int(self.pidx[conf[i], conf[j]])
                if p < self.n2:
                    cols[self.n1 + p] = cols.get(self.n1 + p, 0) + 1
        return cols

    def mono_rows(self):
        """g_2[l, m] - g_2[l+1, m] >= 0 and g_2[l, m] - g_2[l, m+1] >= 0 (zero outside the support)."""
        rows = []
        for p in range(self.n2):
            l, m = int(self.pl[p]), int(self.pm[p])
            for l2, m2 in ((l + 1, m), (l, m + 1)):
                if l2 > m2:
                    l2, m2 = m2, l2
                q = int(self.pidx[l2, m2]) if l2 < self.ncell and m2 < self.ncell else self.n2
                if q < self.n2 and q != p:
                    rows.append({self.n1 + p: 1.0, self.n1 + q: -1.0})
        return rows

    # ---------------------------------------------------------------- the LP by cutting planes
    def optimise(self, k, kp, ncut=200, tol=1e-7, maxit=400, verbose=False):
        side = kp + 1
        joint = [(k1, k + 1 - k1) for k1 in range(1, k + 1) if k1 <= k + 1 - k1 <= kp]
        needed = sorted({side} | {j for pr in joint for j in pr})
        for j in needed:
            self.configs(j)
        rows_i, rows_j, rows_v, rhs = [], [], [], []
        seen = set()

        def add(cols, r):
            key = (tuple(sorted(cols.items())), r)
            if key in seen:
                return False
            seen.add(key)
            idx = len(rhs)
            for c, v in cols.items():
                rows_i.append(idx)
                rows_j.append(c)
                rows_v.append(float(v))
            rhs.append(r)
            return True

        def merge(c1, c2):
            out = dict(c1)
            for c, v in c2.items():
                out[c] = out.get(c, 0) + v
            return out

        nfixed = 0
        if self.mono and self.n2:
            for cols in self.mono_rows():
                add(cols, 0.0)
            nfixed = len(rhs)
        x = np.zeros(self.n)
        for it in range(maxit):
            if len(rhs) > nfixed:
                A = csr_matrix((rows_v, (rows_i, rows_j)), shape=(len(rhs), self.n))
                res = linprog(self.cost, A_ub=-A, b_ub=-np.array(rhs), bounds=[(0, 5)] * self.n, method="highs")
                if res.status != 0:
                    raise RuntimeError(res.message)
                x = res.x
                self.res, self.A = res, A
            pen = {j: self.penalty(x, j) for j in needed}
            new = 0
            P = pen[side]                                                  # side condition G_{kp+1}(1) >= 1
            bad = np.nonzero(P < 1.0 - tol)[0]
            if len(bad):
                for i in bad[np.argsort(P[bad])[:ncut]]:
                    new += add(self.row(self.L[side][i]), 1.0)
            for k1, k2 in joint:                                           # joint conditions G_{k1}(1) + G_{k2}(1) >= 1
                P1, P2 = pen[k1], pen[k2]
                if k1 == k2:
                    bad = np.nonzero(P1 < 0.5 - tol)[0]
                    for i in bad[np.argsort(P1[bad])[:ncut]]:
                        new += add(self.row(self.L[k1][i]), 0.5)
                else:
                    m1, m2 = float(P1.min()), float(P2.min())
                    if m1 + m2 < 1.0 - tol:
                        top1 = np.argsort(P1)[:max(2, ncut // 10)]
                        top2 = np.argsort(P2)[:max(2, ncut // 10)]
                        i1, i2 = int(np.argmin(P1)), int(np.argmin(P2))
                        for i in top1:
                            if P1[i] + m2 < 1.0 - tol:
                                new += add(merge(self.row(self.L[k1][i]), self.row(self.L[k2][i2])), 1.0)
                        for i in top2:
                            if m1 + P2[i] < 1.0 - tol:
                                new += add(merge(self.row(self.L[k1][i1]), self.row(self.L[k2][i])), 1.0)
            if verbose:
                print(f"      round {it}: {len(rhs) - nfixed} cuts, cost {float(self.cost @ x):.5f}, "
                      + ", ".join(f"G_{j} = {float(pen[j].min()):.4f}" for j in needed), flush=True)
            if new == 0:
                return x, float(self.cost @ x), it
        raise RuntimeError("no convergence")

    # ---------------------------------------------------------------- reporting
    def describe(self, x, out):
        g1, g2 = x[:self.n1], x[self.n1:]
        tt = self.tl[:self.n1]
        supp1 = tt[g1 > 1e-9]
        out.append(f"      g_1: cost R_1 = {float(self.cost1 @ g1):.4f}, support [{supp1.min() if len(supp1) else 0:.4f}, "
                   f"{supp1.max() + DELTA if len(supp1) else 0:.4f}); values at sample nodes: "
                   + ", ".join(f"g_1({t:.3f})={v:.3f}" for t, v in
                               [(float(tt[i]), float(g1[i])) for i in range(0, self.n1, max(1, self.n1 // 12))]))
        if self.n2:
            nz = np.nonzero(g2 > 1e-9)[0]
            out.append(f"      g_2: cost R_2 = {float(self.cost2 @ g2):.4f}, {len(nz)} of {self.n2} pair cells used"
                       + (": " + ", ".join(f"g_2({self.tl[self.pl[i]]:.3f},{self.tl[self.pm[i]]:.3f})={g2[i]:.3f}"
                                           for i in nz[np.argsort(-g2[nz])][:24]) if len(nz) else ""))
            if len(nz):
                uv = self.tl[self.pl[nz]] + self.tl[self.pm[nz]]
                out.append(f"      g_2 lives at u+v in [{uv.min():.3f}, {uv.max() + 2 * DELTA:.3f}], "
                           f"i.e. at s = (theta-u-v)/a in [{(theta - uv.max() - 2 * DELTA) / self.a:.2f}, "
                           f"{(theta - uv.min()) / self.a:.2f}]")


def run_pair(out, k, kp, a_values, variants, detail_best=True):
    out.append(f"\npair ({k},{kp}): constraints G_{kp + 1}(1) >= 1 and G_k1(1) + G_k2(1) >= 1 for k1 + k2 = {k + 1}, "
               f"k1, k2 <= {kp}; g_1 free on [a, {theta - 2 * DELTA:.3f}), g_2 on the cell pairs with "
               f"(theta - u - v)/a >= s_min")
    best = {name: None for name, _ in variants}
    for a in a_values:
        L = float(D2.fs(theta / a))
        line = f"   a = {a:.4f} (f_2(theta/a) = {L:.4f}):"
        for name, kw in variants:
            t0 = time.time()
            ps = PairSieve(a, **kw)
            try:
                x, R, its = ps.optimise(k, kp)
            except RuntimeError as exc:
                line += f"  {name}: failed ({exc});"
                continue
            marg = L - R
            R1 = float(ps.cost1 @ x[:ps.n1])
            R2 = float(ps.cost2 @ x[ps.n1:]) if ps.n2 else 0.0
            line += f"  {name}: margin {marg:+.4f} (R_1 = {R1:.4f}, R_2 = {R2:.4f}, {its} rounds, {time.time() - t0:.0f}s);"
            if best[name] is None or marg > best[name][0]:
                best[name] = (marg, a, x, ps)
        out.append(line)
        print(line, flush=True)
    for name, _ in variants:
        if best[name] is None:
            continue
        marg, a, x, ps = best[name]
        out.append(f"   BEST {name} for ({k},{kp}): margin {marg:+.4f} at a = {a:.4f}")
        if detail_best:
            ps.describe(x, out)
            for kk in sorted({kp + 1, kp, kp - 1}):
                val, conf = ps.Gk(x, kk)
                out.append(f"      G_{kk}(1) = {val:.4f} at t = {[round(c, 3) for c in conf]}")
            np.savez(os.path.join(RES, f'optimal_weights_2d_{k}{kp}_{name.split("(")[0]}_{DELTA}.npz'), a=a, margin=marg,
                     tl=ps.tl, n1=ps.n1, g1=x[:ps.n1], pl=ps.pl, pm=ps.pm, g2=x[ps.n1:], cost1=ps.cost1, cost2=ps.cost2)
    return best


if __name__ == "__main__":
    out = [f"theta = {theta}, step {DELTA}; g_1 piecewise linear at the nodes, g_2 constant on cell pairs and non-increasing, "
           f"lambda = mu = 1; margins f_2(theta/a) - R_1(g_1) - R_2(g_2); configurations: all sorted node tuples with "
           f"sum <= 1 (exact for these weights)",
           "F_2 at s = 4, 4.7, 5, 5.4, 6, 6.7, 8: " + ", ".join(f"F_2({s}) = {float(D2.Fs(s)):.4f}"
                                                                for s in (4, 4.7, 5, 5.4, 6, 6.7, 8))]
    def on_grid(vals):
        return [a for a in vals if abs(a / DELTA - round(a / DELTA)) < 1e-9]
    a_values = on_grid([0.05, 0.055, 0.06, 0.065])
    variants = [("single", dict(pairs=False)),
                ("pairs(s>=1)", dict(pairs=True, s_min=1.0))]
    best = run_pair(out, 7, 4, a_values, variants)
    a_star = best["pairs(s>=1)"][1] if best["pairs(s>=1)"] else 0.06
    out.append(f"\nsensitivity at a = {a_star:.4f} for (7,4):")
    for name, kw in [("pairs(s>=0.5)", dict(pairs=True, s_min=0.5)),
                     ("pairs(s>=1), g_2 not monotone (node check only)", dict(pairs=True, s_min=1.0, mono=False)),
                     ("single, F_2=1", dict(pairs=False, flat=True)),
                     ("pairs(s>=1), F_2=1", dict(pairs=True, s_min=1.0, flat=True)),
                     ("pairs(s>=0.5), F_2=1", dict(pairs=True, s_min=0.5, flat=True))]:
        t0 = time.time()
        ps = PairSieve(a_star, **kw)
        x, R, its = ps.optimise(7, 4)
        L = float(D2.fs(theta / a_star))
        line = (f"   {name}: margin {L - R:+.4f} (R_1 = {float(ps.cost1 @ x[:ps.n1]):.4f}, "
                f"R_2 = {float(ps.cost2 @ x[ps.n1:]) if ps.n2 else 0.0:.4f}, {its} rounds, {time.time() - t0:.0f}s)")
        out.append(line)
        print(line, flush=True)
        ps.describe(x, out)
        for kk in (4, 5):
            val, conf = ps.Gk(x, kk)
            out.append(f"      G_{kk}(1) = {val:.4f} at t = {[round(c, 3) for c in conf]}")
    out.append(f"\nhow much cheaper would the pair term have to be?  (7,4) at a = {a_star:.4f}, pair costs x (1 - phi):")
    L = float(D2.fs(theta / a_star))
    for phi in (0.2, 0.4, 0.6, 0.8):
        t0 = time.time()
        ps = PairSieve(a_star, pairs=True, s_min=1.0, discount=phi)
        x, R, its = ps.optimise(7, 4)
        R1 = float(ps.cost1 @ x[:ps.n1])
        R2 = float(ps.cost2 @ x[ps.n1:])
        nz = int(np.sum(x[ps.n1:] > 1e-9))
        line = (f"   phi = {phi:.1f}: hypothetical margin {L - R:+.4f} (R_1 = {R1:.4f}, discounted R_2 = {R2:.4f}, "
                f"{nz} pair cells used, {its} rounds, {time.time() - t0:.0f}s)")
        out.append(line)
        print(line, flush=True)
        if nz:
            ps.describe(x, out)
    run_pair(out, 6, 4, on_grid([0.05, 0.055]), variants, detail_best=False)
    run_pair(out, 7, 5, on_grid([0.055, 0.06])[:1], variants, detail_best=True)
    text = "\n".join(out)
    print(text)
    with open(os.path.join(RES, f'optimal_weights_2d_{theta:.3f}_{DELTA}.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + "\n")
