# Sums of two squares with few prime factors, III: working notes and the level of distribution in short intervals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23826860.svg)](https://doi.org/10.5281/zenodo.23826860)

This repository holds the working notes `notes/part3_explorations.pdf` (20 pages) written after Parts I and II
([two-squares-theta-level](https://github.com/Ruqing1963/two-squares-theta-level), DOI 10.5281/zenodo.23826176;
[two-squares-mean-square](https://github.com/Ruqing1963/two-squares-mean-square), DOI 10.5281/zenodo.23826297), the
scripts behind them, and their outputs. Sections 1, 3, 5
and the conclusions of Sections 2, 4 and 6 have been incorporated into Part II (Sections 7.3-7.5 and 8.1 there, where
the notes are also mirrored in `notes/`); Section 7, the level of distribution in short intervals, is new here:

* **Theorem 7.1 (unconditional).** For H = X^Δ with Δ > 1/2, all but O(H X^{-ε/4}) of the N ≡ 2 (mod 4) in
  (X, X+H] have the level of distribution Δ − 3ε for r(n) r(N − n); hence (6,4) for almost all N in every interval of
  length X^{2/3+ε} and (5,4) in every interval of length X^{19/20+ε}.
* **Proposition 7.2 (conditional).** The short-interval mean square Σ_{X<n≤X+H} |a_G(n)|² ≪ X^η (H + L) X ‖G‖²/L
  would raise the level to (1 + Δ)/2 for every Δ > 0. This is a large-sieve inequality for the Poincaré series of the
  interval, λ_max ≪ (1 + H/L)/X, sharp, and its Kloosterman form is the obstacle of Section 2.4 in its symmetric shape.
  `short_interval_mean_square.py` tests it on the pairs of Part II (coefficients exact to 2^21, all windows with
  X in [2^19, 2^20]): the coefficient mass of G_{m,m'} is spread at every scale down to H = 16, independently of L
  (the maximal window ratio is 4-6.5 at H = 16 and 1.0-1.1 at H = 2^16 for L from 20 to 41612), so the hypothesis
  holds for these forms with room to spare and without the term L. Conjecture 7.3 states this bound for the theta
  forms (at H = 1 it is the individual conjecture of Part II, at H = X the mean-square theorem), and Corollary 7.4
  shows that it would give the level 1 − ε for almost all N in every interval of length X^Δ, Δ > 0.
* The notes also record what is NOT proved: the trilinear form of Section 2 is the leading term (the terms g > 1, the
  ternary term and the smoothing error are not carried out), the Poisson analysis of Section 4 is bookkeeping under the
  most favourable assumption, and the optimality of Richert's weight in Section 6 is numerical (an LP on a grid), not
  a theorem.

Working notes and scripts around the level of distribution 1/2 of r(n) r(N-n) (Part I, Theorem 3.2; Part II,
Section 7), written after Parts I and II were finalised. Nothing here is part of either paper.

* `notes/part3_explorations.tex` (+ PDF): three explorations.
  1. **The symmetric remainder** R_d = sum_{d1 d2 = d} R(d1, d2; N) of the two-dimensional sieve. Lemma 1.1 moves any
     factor of m across Theta_{m,m'} by an Atkin-Lehner involution; Proposition 1.2 gives
     sum_{u | d} G_{t u, d/u} = prod_{p | d} (1 + W_p) G_{t,d}, so the sieve sees only the joint (+1)-eigenspace of the
     W_p and every Hecke component with eta_p(f) = -1 for some p | (d, M_1) cancels exactly (all oldforms of odd level,
     Proposition 1.4). Numerically the cancellation is of square-root strength (0.67-0.69 of the absolute sum for
     omega(d) = 2, 0.46-0.48 for omega(d) = 3; random signs: 0.707, 0.5) and a d^{o(1)} gain: it does not move the level.
  2. **The dispersion form of the remainder sum.** The Fourier transform of the circle x^2 + y^2 = a mod c is a
     classical Kloosterman sum (Lemma 2.1, checked numerically); the remainder a_G(N) - a_G(N/2) is written exactly as a
     sum of S(16N, m|h|^2; m'k) over the moduli c = m'k = 0 (mod m'), c <= sqrt(N m') (Proposition 2.2); the character
     chi_{-4} disappears. The uniform bound of Part I is equivalent to square-root cancellation in the sum over k; the
     Weil bound alone gives the level 1/4, square-root cancellation over m' would give 2/3, over m as well 1. No level
     above 1/2 is proved; the three obstacles for the Deshouillers-Iwaniec / BFI machinery are listed.
  3. **The dimension of the weighted sieve.** The factor 2^Omega(n(N-n)) only affects the constant; the weights
     r(n) r(N-n) cost the sieve dimension (2 instead of 1). Benchmark with the same Richert weights: dimension 1 would
     give (5,3) at theta = 1/2 and (3,2), i.e. N = P_1 + P_2, from theta = 4/5, against (7,5) and (5,3); r(pn') = 2r(n')
     does not lower the dimension and the switching trick leads back to problem 2.
  4. **Cauchy-Schwarz and Poisson against the fixed first argument 16N.** Exponent bookkeeping for both Cauchy-Schwarz
     arrangements of the trilinear form and for the Poisson dual in h: the h-box has side <= sqrt(mm') while the modulus
     of the Poisson step is >= sqrt(N), so the dual is longer by N^{1/2 - theta/4}; the first arrangement's diagonal
     alone gives at most the level 2/5. Negative.
  5. **Mollified theta weights.** r(n) (sum_{u|n, u<=N^xi} g_u)^2 keeps the theta structure at the level 1/2 - 4 xi;
     Theorem 5.1: for xi < a the mollifier is identically 1 on the sifted integers (same sieve, lower level);
     Proposition 5.2: the density at p changes only if the mollifier contains p, so acting on the Richert range costs
     4b (two-sided) or 2b (one-sided) in level; the hybrid sieve is worse than (7,5) for every xi >= a (8.57/5.60 and
     8.04/5.33 against 7.56/5.07).
  6. **The weight function at the level 1/2.** A linear programme over all non-negative weight functions g on [a, b)
     (constraints G_{k'+1}(1) >= 1, G_{(k+1)/2}(1) >= 1/2 with G_k(S) = inf sum g(t_i) over the simplex, evaluated by
     dynamic programming and added as cutting planes) shows that the optimal g is the linear Richert weight itself:
     the 0.030 gap to (7,4) is structural. The vector-sieve upper bound F_1 F_1 never beats F_2, and an explicit
     subtraction of the five-prime integers lies outside the level. Section 2.5 substitutes Drappeau's Theorem 2.1
     and the Kuznetsov/large-sieve route into the trilinear form: in every assignment the single first argument 16N
     is the term that blocks theta > 1/2.
* `code/` (the scripts import the modules of Part II from `../two-squares-mean-square/code`):
  `symmetric_remainder.py` (all R(d1,d2;N) exactly, 24 values of N near 2e6, ratios and the empirical eigenspace share),
  `symmetric_norm_share.py` (Rankin-Selberg shares of the joint Atkin-Lehner eigenspaces of G_{1,d}, also restricted to the
  residue classes the sieve sees), `kloosterman_check.py` (Lemma 2.1 for prime and composite square-free moduli),
  `dimension_benchmark.py` (Richert thresholds for kappa = 1 and kappa = 2), `kloosterman_correlation.py` (sizes and
  correlations of the Kloosterman sums of Section 4), `mollified_benchmark.py` (hybrid sieve thresholds of Section 5,
  including the DHR constants for kappa = 3/2), `optimal_weights.py` (LP over the weight function, Section 6).
* `results/`: the outputs of the seven scripts.
* The positive results (Sections 1, 3.2, 5.1-5.2, 2.1-2.2 and the bookkeeping) have been integrated into Part II
  (Sections 7.3, 7.5 and 8.1 there).

Floating point throughout; evidence, not proof.
