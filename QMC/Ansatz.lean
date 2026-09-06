import SSA
import Mathlib.Analysis.Calculus.ContDiff.Basic

/-!
# `Ansatz.lean` — the validity contract for AI-generated QMC ansätze

An ansatz written in the SSA/XLA DSL is *valid* if its mathematical semantics
(`Xla.simpleEval`) satisfies `IsValidQMCWavefunction` below. The contract has exactly
two fields; this docstring explains why two suffice, and where everything else went.

## Design principle: prove only what cheats

A requirement earns its per-candidate proof cost only if violating it yields a
*spuriously low* energy that Monte Carlo converges to **silently** — poisoning the
ansatz search. A requirement whose violation merely raises the energy, or makes Monte
Carlo diverge loudly (NaN, drift), is self-punishing: the fitness function rejects the
offender on its own. Those are policed by cheap runtime monitors (`pfnet/monitor.py`),
not by proofs. Under this criterion exactly two conditions survive.

## Cheat 1 — broken antisymmetry: bosonic collapse

Without antisymmetry, Monte Carlo converges cleanly to the Rayleigh quotient of a
non-fermionic wavefunction, floored only by the bosonic ground energy
`E₀^bosonic < E₀^fermionic`: low variance, no warning, wrong physics. Both spin
sectors must be checked — a mixed-symmetry ansatz still cheats.

## Cheat 2 — broken smoothness: AD drops distributional deltas

`pfnet` evaluates the local energy `E_L = -½(Δ log Ψ + |∇ log Ψ|²) + V` by automatic
differentiation (`pfnet/qmc.py:make_E_local`), which sees only the *regular* part of
the Laplacian. Write the distributional Laplacian as `ΔΨ = f + μ` with `f` a locally
integrable function and `μ` a singular measure. Pairing with `Ψ` and comparing
against the kinetic quadratic form `½∫|∇Ψ|²` gives the estimator bias

```
E_MC = E_true + ½ ∫ Ψ dμ / ∫ Ψ².
```

The cheat bias *is* the singular part of `ΔΨ`. At a crease — a codimension-1 surface
`Σ` where `∇Ψ` jumps — `μ = [∂ₙΨ]·dS|_Σ` contributes `½∫_Σ Ψ[∂ₙΨ] dS / ∫Ψ²`, which is
first-order in the crease depth while the true energy cost is second-order: the
optimizer reliably finds it, and the result is floored by nothing (it is not any
variational object's energy). Value jumps are worse: true kinetic energy `+∞`,
estimator finite.

### The exact requirement (for the record)

`smooth` as stated (C²) is the provable, compositional sufficient condition. The
*exact* requirement is:

* `Ψ ∈ W^{2,1}_loc` — the distributional Laplacian has no singular part. This
  excludes value jumps and creases but *includes* Kato cusps: `e^{-Zr}` has bounded
  gradient and `ΔΨ ~ 1/r`, locally integrable in 3D, so no delta forms;
* `Ψ` twice differentiable at `Ψ²`-almost-every point, so the AD-evaluated `E_L`
  equals the formula at sample points (automatic for piecewise-analytic DSL
  programs);
* `∫ Ψ dμ = 0` wherever a singular part is present — creases on the *nodal set* are
  harmless, because `Ψ = 0` there.

`C² ⊂ C¹ + piecewise C² ⊂ W^{2,1}_loc = exact`. `W^{2,1}_loc` is not formalized in
mathlib, and C² is what is statable and provable compositionally; if the search ever
rediscovers exact-cusp factors, relax `smooth` to "piecewise C² with `ΔΨ ∈ L¹_loc`
across the interfaces".

### LLM-facing op discipline (how candidates satisfy `smooth`)

* always safe (real-analytic): `add`, `mul`, `sub`, `neg`, `exp`, `sum`,
  `transpose`, `broadcast`, `einsum`, `dot_general`, `det`;
* `sqrt`: analytic on `(0, ∞)`; keep the argument away from 0 (`sqrt(x²+ε)`), or
  cancel it (`abs(x)^2 = x²`). `sqrt(x²) = |x|` is a crease;
* `abs`: crease at 0; safe only if the composition is even in that argument, or the
  zero set is provably nodal;
* `choice` on float predicates: crease at the boundary unless the branches meet with
  continuous gradient (the smooth-cutoff idiom), or the boundary is nodal;
* `mod`, `gather`, `iota`: piecewise constant in continuous arguments — only on
  integer data (`Z_nuc`, indices), never on electron coordinates.

## Not in the structure, and why (monitored in `pfnet/monitor.py` instead)

* **`Ψ ∈ L²`**: at infinite time self-punishing (no stationary distribution), but
  *finite* runs fake convergence — walkers drift to infinity under a growing `Ψ`
  (`E_L → -∞` quadratically), or collapse onto a non-integrable spike and sit in
  long "settled" epochs. Monitored: walker-radius drift, walker-spread collapse,
  energy stationarity.
* **`∇Ψ ∈ L²`, `ΨΔΨ ∈ L¹`**: true energy `+∞`, or the `E_L` mean fails to settle;
  trips the local-energy finiteness check or the grad-clip assert
  (`pfnet/qmc.py:update`).
* **vanishing at infinity**: the variational principle for `H¹` functions never
  needs it; spikes at infinity carry vanishing sampling mass.
* **θ-differentiability**: bad parameter gradients only slow optimization; energy
  evaluation is unaffected.
* **nodes of positive measure / flat zero regions**: the sampler never accepts moves
  into zero-density regions; the restricted `Ψ` is still `H¹`, still `≥ E₀`.
* **Kato cusp**: heavy `E_L` tails (divergent higher moments) hurt convergence
  *rate* — efficiency, not correctness of the mean.
* **`Ψ ≢ 0`**: immediate NaN in the acceptance ratio; caught by the log-amplitude
  check.

## The `∀θ` contract: reparameterize, don't restrict

`Ansatz.isValid` quantifies over *all* parameter values, because `pfnet` optimizes
with unconstrained SGD (`pfnet/qmc.py:update`) — there is no parameter domain to
enforce. Consequence: every quantity that must stay positive (orbital exponents,
widths, cutoff scales) must be reparameterized inside the ansatz itself via `exp`,
square, or softplus.

## Boundary with `pfnet`: log-amplitude

The DSL ansatz outputs the amplitude `Ψ`; `pfnet` consumes log-amplitude
(`log_psi(param, r, R, Z)`; `pfnet/model.py` returns `log|det| + log` envelope). The
export wrapper is `log_psi(x) = log |Ψ(x)|`, packing spin-up electrons first in `r`.
Off the nodal set (which has `Ψ²`-measure zero),
`Δ log|Ψ| + |∇ log|Ψ||² = ΔΨ/Ψ`, so `make_E_local` evaluates `HΨ/Ψ`, and the
Metropolis sampler (`pfnet/qmc.py:make_sampler`) targets `p(x) ∝ |Ψ(x)|²`.
-/

/-! ## Specification -/

/-- Wavefunction of a molecule with `n₁` spin-up and `n₂` spin-down electrons:
`x₁ i`, `x₂ j : Fin 3 → ℝ` are electron positions, the value is the (real) amplitude. -/
abbrev Wavefunction (n₁ n₂ : ℕ) := (Fin n₁ → Fin 3 → ℝ) → (Fin n₂ → Fin 3 → ℝ) → ℝ

/-- Pauli antisymmetry within each spin sector: permuting the positions of same-spin
electrons multiplies `Ψ` by the sign of the permutation. (Up/down exchange is not a
symmetry of the spin-labeled wavefunction.) -/
def IsAntisymmetric {n₁ n₂ : ℕ} (Ψ : Wavefunction n₁ n₂) : Prop :=
  ∀ (σ₁ : Equiv.Perm (Fin n₁)) (σ₂ : Equiv.Perm (Fin n₂)), ∀ x₁ x₂,
    Ψ (x₁ ∘ σ₁) (x₂ ∘ σ₂) = σ₁.sign • σ₂.sign • Ψ x₁ x₂

/-- The validity contract: exactly the two properties whose violation is a *silent*
cheat that Monte Carlo converges to (bosonic collapse; AD dropping distributional
deltas). Everything else is self-punishing under Monte Carlo and is monitored at
runtime in `pfnet/monitor.py`. See the module docstring for the full analysis. -/
structure IsValidQMCWavefunction {n₁ n₂ : ℕ} (Ψ : Wavefunction n₁ n₂) : Prop where
  /-- Pauli principle in both spin sectors: without it, sampling silently converges
  to a bosonic/mixed-symmetry energy below the fermionic ground state. -/
  antisymmetric : IsAntisymmetric Ψ
  /-- No jumps or creases: automatic differentiation computes the true Laplacian,
  so the local-energy estimator is unbiased. (The exact condition is that the
  distributional `ΔΨ` has no singular part, i.e. `Ψ ∈ W^{2,1}_loc`; C² is the
  provable sufficient condition.) -/
  smooth : ContDiff ℝ 2 (Function.uncurry Ψ)

/-- Wavefunction ansatz. The parameter is a 1D vector. An ansatz should support different
numbers of atoms and electrons -/
structure Ansatz where
  N_param : ℕ
  ansatz (N_nuc N_up N_down : ℕ) :
    SSA.Expr Xla.XlaOp
      [
        ⟨.float, [N_param]⟩, -- parameters
        ⟨.float, [N_nuc, 3]⟩, -- positions of nuclei
        ⟨.int, [N_nuc]⟩, -- types of nuclei
        ⟨.float, [N_up, 3]⟩, -- positions of spin up electrons
        ⟨.float, [N_down, 3]⟩ -- positions of spin down electrons
      ]
      [⟨.float, []⟩]

/-- An ansatz is valid if for any param and any molecule, the wavefunction is valid. -/
def Ansatz.isValid (ansatz : Ansatz) : Prop :=
  ∀ (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin ansatz.N_param → ℝ)
    (R_nuc : SSA.Tensor ℝ [N_nuc, 3])
    (Z_nuc : SSA.Tensor ℤ [N_nuc]),
  IsValidQMCWavefunction <|
    Xla.simpleEval (ansatz.ansatz N_nuc N_up N_down) θ R_nuc Z_nuc
