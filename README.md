# Sums of two squares with few prime factors, III: the dispersion obstacle, sieve barriers at the level 1/2, and short-interval mean squares

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23826860.svg)](https://doi.org/10.5281/zenodo.23826860)

Part III of the series after Part I ([two-squares-theta-level](https://github.com/Ruqing1963/two-squares-theta-level),
DOI 10.5281/zenodo.23826176) and Part II ([two-squares-mean-square](https://github.com/Ruqing1963/two-squares-mean-square),
DOI 10.5281/zenodo.23826297). The repository holds

* `paper/two_squares_part3.tex` (+ PDF, 32 pages, 4 figures): the paper;
* `notes/part3_explorations.tex` (+ PDF): the working notes from which the paper grew (also mirrored in Part II, `notes/`);
* `code/`: the scripts (they import the modules of Part II from `../two-squares-mean-square/code`);
* `results/`: their outputs; `figures/`: the four figures (PDF and PNG).

## The paper

Parts I and II proved the level of distribution 1/2 for r(n) r(N − n) through the theta series θ²(mz)θ²(m'z), and with it
Ω(n(N − n)) ≤ 7, max(Ω(n), Ω(N − n)) ≤ 5 for every large N ≡ 2 (mod 4). Part III examines the three questions these
results leave open.

1. **The symmetric remainder (Section 2).** Lemma 2.1 moves any part of the parameter m of Θ_{m,m'} across an
   Atkin–Lehner involution; Proposition 2.2 gives Σ_{u|d} G_{t u, d/u} = 2^{ω(d)} P⁺ G_{t,d}, the projection onto the
   joint (+1)-eigenspace of the W_p, p | d. So the remainder R_d = Σ_{d₁d₂=d} R(d₁,d₂;N) of the two-dimensional sieve
   sees only that eigenspace, and every Hecke component with η_p(f) = −1 for some p | (d, M₁) cancels exactly, in
   particular every oldform of odd level (Proposition 2.4). Numerically the cancellation is of square-root strength
   (0.67–0.69 of the absolute sum for ω(d) = 2, 0.46–0.48 for ω(d) = 3; random signs 0.707, 0.5): a d^{o(1)} gain that
   does not move the level (Table 1, Figure 1).
2. **The dispersion form (Sections 3–4).** The Fourier transform of the circle x² + y² ≡ a modulo any odd c is a
   classical Kloosterman sum (Lemma 3.1, verified numerically), and a_G(N) − a_G(N/2) has as its leading term a sum of
   S(16N, m|h|²; m'k) over the moduli c ≡ 0 (mod m'), the character χ_{−4} cancelling against the Gauss sums
   (Proposition 3.2). The uniform bound of Part I is square-root cancellation over k; over m' it would give the level
   2/3, over m as well 1; the Weil bound gives 1/4. Substituting the Deshouillers–Iwaniec and Drappeau bounds in every
   assignment (Section 3.5) and applying Cauchy–Schwarz with Poisson in the dual variable h (Section 4) finds the same
   obstruction each time: the single fixed first argument 16N (the h-box has side ≤ √(mm') while every modulus is
   ≥ √N). Figure 2 shows the measured correlations of the h-sums.
3. **Sieve barriers at the level 1/2 (Sections 5–6).** The factor 2^Ω enters only the constant; what the weights
   r(n) r(N − n) cost is the dimension, 2 against 1 (a dimension-one sieve would give (5,3) at 1/2 and (3,2) from 4/5).
   Mollifiers truncated below the sifting limit are identically 1 on the sifted integers (Theorem 5.1); mollifiers
   reaching the Richert range change the density only at the primes they contain (Proposition 5.2) and cost more level
   than they return (8.57/5.60 and 8.04/5.33 against 7.56/5.07). A linear programme over all non-negative weight
   functions, with the prime-configuration constraints evaluated by dynamic programming, finds Richert's linear weight
   optimal on the grid for (7,5), (7,4), (6,4): the 0.030 gap to (7,4) is structural (Figure 3).
4. **Short intervals (Sections 7–8).** Theorem 7.1 (unconditional): for H = X^Δ, Δ > 1/2, almost all N ≡ 2 (mod 4) in
   (X, X + H] have the level Δ − 3ε, hence (6,4) in every interval of length X^{2/3+ε} and (5,4) in every interval of
   length X^{19/20+ε}. Proposition 7.2: the short-interval mean square Σ_{X<n≤X+H} |a_G(n)|² ≪ X^η (H + L) X ‖G‖²/L
   for all G of level L would give (1 + Δ)/2 for every Δ > 0; Conjecture 7.3 is the same bound without the term L for
   the theta forms alone, interpolating between the individual conjecture of Part II (H = 1) and its mean-square theorem
   (H = X), and Corollary 7.4 derives from it the level 1 − ε for almost all N in every interval of length X^Δ. Section 8
   explains the term L spectrally. Theorem 8.1 (unconditional): the diagonal of the Hecke decomposition satisfies the
   conjectured bound X^ε HX‖G‖²/L for every H ≥ X^ε (Deligne's bound, Shiu's theorem on Σ d(n)² over short intervals,
   Hoffstein–Lockhart for the Petersson norms, Gram matrices of the oldforms), so Conjecture 7.3 is equivalent to the
   cancellation of the off-diagonal (Corollary 8.2), and the trivial spectral bound, which at H = 1 is the uniform bound
   of Part I, exceeds it by exactly the factor L (Remark 8.3). The term L is the coherence of the Poincaré series,
   which aligns all spectral phases at one index, while the theta forms have window-independent coefficients (central
   values) whose off-diagonal correlations average out (Proposition 8.4). The data (Table 2, Figure 4) show the
   off-diagonal of G_{1,q} averaging to zero with root mean square decreasing like H^{−1/2}, and the window ratios of
   G_{m,m'} independent of L down to H = 16. Proposition 8.5: the lattice-point form of the variance (zeros of
   Q ⊕ (−Q) in eight variables in a shell) does not avoid the spectral problem: it needs a twisted first moment of a_G
   with square-root cancellation, and the precision HX/L lies below the floor X²‖G‖²/L of every circle-method error
   term.
5. **The bilinear form in the two moduli (Section 4.5).** The Chinese remainder theorem factors S(16N, n; m'k) and
   writes the remainder as a bilinear form in (k, m') in which the fixed argument enters modulo m' as 16N k̄; the mixing
   of k̄ is real and already used by the theta bound, so a saving must be bilinear (as in BFI, DFI). Section 6.4 (end):
   the LP covers separable weights only; pair weights lead to a larger LP with the same inputs, Chen's switching to the
   remainder problem of Section 3 with a prime variable inside the modulus.

What is proved: Lemma 2.1, Propositions 2.2, 2.4, Lemma 3.1, Proposition 3.2, Theorems 5.1, 7.1 and 8.1, Propositions
5.2, 7.2, 8.4, 8.5, Corollaries 7.4 and 8.2 (7.2 and 7.4 conditional). What is not: the exponent bookkeeping of
Sections 3.3–3.5 and 4 concerns the leading term of Proposition 3.2 under the most favourable assumptions; the
optimality of Richert's weight is the solution of an LP on a grid; the comparisons with random signs, the eigenspace
shares and the short-interval ratios are numerical observations.

## Scripts (`code/`)

`symmetric_remainder.py`, `symmetric_norm_share.py` (Section 2.4), `kloosterman_check.py` (Lemma 3.1),
`kloosterman_correlation.py` (Section 4), `dimension_benchmark.py`, `simultaneous_pairs.py`, `mollified_benchmark.py`
(Section 5), `optimal_weights.py` (Section 6), `short_interval_mean_square.py`, `short_interval_spectral.py`
(Sections 7–8), and the plotting scripts `plot_symmetric_remainder.py`, `plot_kloosterman_correlation.py`,
`plot_sieve_barriers.py`, `plot_short_interval.py` (Figures 1–4). Outputs in `results/`.

Floating point throughout; evidence, not proof. The theorems do not depend on the computations.
