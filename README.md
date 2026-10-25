# Notes towards Part III

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
* `code/` (the scripts import the modules of Part II from `../two-squares-mean-square/code`):
  `symmetric_remainder.py` (all R(d1,d2;N) exactly, 24 values of N near 2e6, ratios and the empirical eigenspace share),
  `symmetric_norm_share.py` (Rankin-Selberg shares of the joint Atkin-Lehner eigenspaces of G_{1,d}, also restricted to the
  residue classes the sieve sees), `kloosterman_check.py` (Lemma 2.1 for prime and composite square-free moduli),
  `dimension_benchmark.py` (Richert thresholds for kappa = 1 and kappa = 2).
* `results/`: the outputs of the four scripts.

Floating point throughout; evidence, not proof.
