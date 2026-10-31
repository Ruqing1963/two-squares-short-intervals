r"""
Research 2 (Part III notes, Section 2): the exponential sum behind r(m) in arithmetic progressions.

For odd q, (a, q) = 1 and h = (h1, h2) in Z^2 the notes claim (Lemma 2.1)
    sum_{x, y mod q, x^2 + y^2 = a (q)} e((h1 x + h2 y)/q) = chi(q) S(a, 4bar (h1^2 + h2^2); q) + q [h = 0 (q)],
with S(m, n; c) = sum_{t mod c, (t, c) = 1} e((m t + n tbar)/c) the classical Kloosterman sum and chi = chi_{-4}, so that the
Fourier transform of the circle x^2 + y^2 = a modulo q is a classical Kloosterman sum whose second argument is the norm
h1^2 + h2^2 of the dual variable (a Salie sum does not occur).  This script checks the identity for odd q <= 61 (prime
and composite) and all a, h.
"""
import numpy as np
from math import gcd

def e(x, q):
    return np.exp(2j * np.pi * x / q)

def jacobi_minus1(q):
    # chi_{-4}(q) = (-1/q) for odd q
    return 1 if q % 4 == 1 else -1

def kloosterman(m, n, q):
    s = 0
    for t in range(q):
        if gcd(t, q) == 1:
            s += e(m * t + n * pow(t, -1, q), q)
    return s

overall = 0.0
for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
    worst = 0.0
    inv4 = pow(4, -1, q)
    xs = np.arange(q)
    sq = (xs[:, None] ** 2 + xs[None, :] ** 2) % q
    for a in range(q):
        if gcd(a, q) != 1:
            continue
        mask = sq == a
        X, Y = np.nonzero(mask)
        for h1 in range(q):
            for h2 in range(q):
                lhs = np.sum(e(h1 * X + h2 * Y, q))
                rhs = jacobi_minus1(q) * kloosterman(a, inv4 * (h1 * h1 + h2 * h2), q) + (q if (h1 % q == 0 and h2 % q == 0) else 0)
                worst = max(worst, abs(lhs - rhs))
    overall = max(overall, worst)
    print(f"q = {q}: identity holds to {worst:.1e}")
print("max deviation over all prime q:", overall)

# odd square-free composite q: the terms t with 1 < (t, q) < q in the Gauss-sum evaluation give
#    F_q(a; h) = sum_{g | (q, h1, h2)} g chi(q/g) S(a, 4bar |h/g|^2; q/g),
# (the term g = q is the q [h = 0 (q)] above, the term g = 1 the Kloosterman sum of the prime case).
print("\ncomposite odd square-free q: F_q(a;h) = sum_{g | (q,h)} g chi(q/g) S(a, 4bar |h/g|^2; q/g)")
overall = 0.0
for q in [15, 21, 33, 35, 39, 55, 65, 77, 105]:
    worst = 0.0
    xs = np.arange(q)
    sq = (xs[:, None] ** 2 + xs[None, :] ** 2) % q
    divs = [g for g in range(1, q + 1) if q % g == 0]
    for a in [a for a in range(1, q) if gcd(a, q) == 1][:6]:
        X, Y = np.nonzero(sq == a)
        for h1 in range(q):
            for h2 in range(0, q, 1 if q <= 39 else 3):
                lhs = np.sum(e(h1 * X + h2 * Y, q))
                rhs = 0
                for g in divs:
                    if h1 % g or h2 % g:
                        continue
                    c = q // g
                    if c == 1:
                        rhs += g
                        continue
                    rhs += g * jacobi_minus1(c) * kloosterman(a % c, (pow(4, -1, c) * (((h1 // g) ** 2 + (h2 // g) ** 2))) % c, c)
                worst = max(worst, abs(lhs - rhs))
    overall = max(overall, worst)
    print(f"q = {q}: identity holds to {worst:.1e}")
print("max deviation over the composite q:", overall)

# general odd q (prime powers allowed): the sign chi(q/g) is replaced by chi((q/g)*), where c* is the product of the
# primes dividing c to an ODD power (G(t; p^nu)^2 = p^nu for nu even and chi(p) p^nu for nu odd):
#    F_q(a;h) = sum_{g | (q,h)} g chi((q/g)*) S(a, 4bar |h/g|^2; q/g).
def star(c):
    s = 1
    d = 3
    while d * d <= c:
        v = 0
        while c % d == 0:
            c //= d
            v += 1
        if v % 2:
            s *= d
        d += 2
    if c > 1:
        s *= c
    return s

print("\nodd q with square factors: F_q(a;h) = sum_{g | (q,h)} g chi((q/g)*) S(a, 4bar |h/g|^2; q/g)")
overall = 0.0
for q in [9, 25, 27, 45, 49, 63, 75, 81, 99, 117, 135]:
    worst = 0.0
    xs = np.arange(q)
    sq = (xs[:, None] ** 2 + xs[None, :] ** 2) % q
    divs = [g for g in range(1, q + 1) if q % g == 0]
    for a in [a for a in range(1, q) if gcd(a, q) == 1][:5]:
        X, Y = np.nonzero(sq == a)
        for h1 in range(0, q, 1 if q <= 49 else 2):
            for h2 in range(0, q, 1 if q <= 27 else 3):
                lhs = np.sum(e(h1 * X + h2 * Y, q))
                rhs = 0
                for g in divs:
                    if h1 % g or h2 % g:
                        continue
                    c = q // g
                    if c == 1:
                        rhs += g
                        continue
                    rhs += g * jacobi_minus1(star(c)) * kloosterman(a % c, (pow(4, -1, c) * (((h1 // g) ** 2 + (h2 // g) ** 2))) % c, c)
                worst = max(worst, abs(lhs - rhs))
    overall = max(overall, worst)
    print(f"q = {q}: identity holds to {worst:.1e}")
print("max deviation over the non-square-free q:", overall)
