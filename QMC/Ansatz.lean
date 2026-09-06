import SSA
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.MeasureTheory.Measure.Haar.OfBasis

/-!
# `Ansatz.lean` — what a molecular wavefunction must satisfy, and a simple verified ansatz

This file is the contract for AI-generated QMC ansätze: an ansatz written in the
SSA/XLA DSL is *valid* if its mathematical semantics (`Xla.simpleEval`) satisfies
`IsValidQMCWavefunction` below.

The conditions are exactly what Pfau-style variational Monte Carlo
(FermiNet/PauliNet lineage, cf. `pfnet`) needs to give correct energies:

* **Antisymmetry within each spin sector** (`antisymmetric`): electrons are fermions.
  Swapping two same-spin electrons must negate `Ψ` (Pauli principle).
* **C² smoothness** (`smooth`): the local energy
  `E_L(x) = HΨ/Ψ = -½(Δ log Ψ + |∇ log Ψ|²) + V(x)` (the form evaluated in
  `pfnet/qmc.py:make_E_local`) requires the Laplacian of `Ψ` to exist.
  C², not C^∞: practical ansätze use finitely-smooth cutoffs (cf. the TODO in
  `pfnet/model.py`), so the spec does not require more.
* **Square-integrability** (`sq_integrable`): the Metropolis sampler
  (`pfnet/qmc.py:make_sampler`) draws configurations from `p(x) ∝ |Ψ(x)|²`;
  this is a probability density only if `Ψ²` is integrable.
* **Vanishing at infinity** (`vanishes_at_infinity`): the bound-state boundary
  condition, needed so the integration by parts behind the variational principle
  `E[Ψ] ≥ E₀` has no boundary terms. NOTE: this is *not* implied by
  square-integrability (a continuous L² function can have unit-height spikes of
  shrinking width marching off to infinity), so it is stated separately.

Deliberately not fields of the structure:

* **Nonzeroness** (`∃ x, Ψ x ≠ 0`): needed to normalize `|Ψ|²`; carried as an
  explicit hypothesis in the corollaries that need it.
* **Finite kinetic energy** (`∇Ψ ∈ L²`): required for the Rayleigh quotient to be
  finite; future work.
* **Kato cusp conditions**: they keep the local energy bounded at coalescence
  (finite variance / efficiency — `pfnet/model.py` adds an explicit cusp envelope),
  but are not needed for the *mean* energy to be correct.
* **Real-valuedness**: automatic — `DirectImpl` models floats as `ℝ`.
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

/-- The analytical conditions for Pfau-style VMC to give correct energies.
See the module docstring for the justification of each field. -/
structure IsValidQMCWavefunction {n₁ n₂ : ℕ} (Ψ : Wavefunction n₁ n₂) : Prop where
  /-- Pauli principle. -/
  antisymmetric : IsAntisymmetric Ψ
  /-- The Laplacian exists, so the local energy `E_L = HΨ/Ψ` is well-defined. -/
  smooth : ContDiff ℝ 2 (Function.uncurry Ψ)
  /-- `|Ψ|²` can be normalized to the probability density the sampler targets. -/
  sq_integrable : MeasureTheory.Integrable
    (fun x : (Fin n₁ → Fin 3 → ℝ) × (Fin n₂ → Fin 3 → ℝ) ↦ (Ψ x.1 x.2) ^ 2)
  /-- Bound-state boundary condition (not implied by `sq_integrable`). -/
  vanishes_at_infinity : Filter.Tendsto (Function.uncurry Ψ) (Filter.cocompact _) (nhds 0)

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
