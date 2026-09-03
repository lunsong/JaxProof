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

Proofs are currently `sorry` (work in progress); all definitions and statements
are final and compile. The `det` XLA primitive used below lives in
`SSA/Xla/Op.lean` (semantics `Matrix.det` in `SSA/Xla/Impl.lean`, DSL helper in
`SSA/Xla/Libs.lean`, JAX handler in `python/ops.py`).
-/

/-- Redeclared per example-file convention (as in `Coulomb.lean`, `offDiag.lean`, ...). -/
theorem Index.single_zero {ι : Type} {m : ι → Type} {i : ι} {x : m i} : Index.single x 0 = x := rfl

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

namespace IsAntisymmetric

variable {n₁ n₂ : ℕ} {Ψ : Wavefunction n₁ n₂}

/-- Swapping two spin-up electrons negates `Ψ`. -/
theorem swap_up (hΨ : IsAntisymmetric Ψ) (i j : Fin n₁)
    (x₁ : Fin n₁ → Fin 3 → ℝ) (x₂ : Fin n₂ → Fin 3 → ℝ) :
    Ψ (x₁ ∘ Equiv.swap i j) x₂ = -Ψ x₁ x₂ := by
  sorry

/-- Swapping two spin-down electrons negates `Ψ`. -/
theorem swap_down (hΨ : IsAntisymmetric Ψ) (i j : Fin n₂)
    (x₁ : Fin n₁ → Fin 3 → ℝ) (x₂ : Fin n₂ → Fin 3 → ℝ) :
    Ψ x₁ (x₂ ∘ Equiv.swap i j) = -Ψ x₁ x₂ := by
  sorry

/-- Pauli exclusion: two coincident same-spin (up) electrons force `Ψ = 0`. -/
theorem eq_zero_of_coincident_up (hΨ : IsAntisymmetric Ψ) {i j : Fin n₁} (hij : i ≠ j)
    {x₁ : Fin n₁ → Fin 3 → ℝ} (hx : x₁ i = x₁ j) (x₂ : Fin n₂ → Fin 3 → ℝ) :
    Ψ x₁ x₂ = 0 := by
  sorry

/-- Pauli exclusion: two coincident same-spin (down) electrons force `Ψ = 0`.
(Opposite-spin electrons may coincide freely.) -/
theorem eq_zero_of_coincident_down (hΨ : IsAntisymmetric Ψ) {i j : Fin n₂} (hij : i ≠ j)
    (x₁ : Fin n₁ → Fin 3 → ℝ) {x₂ : Fin n₂ → Fin 3 → ℝ} (hx : x₂ i = x₂ j) :
    Ψ x₁ x₂ = 0 := by
  sorry

end IsAntisymmetric

/-! ## The simple ansatz: product of two Gaussian Slater determinants -/

/-- Gaussian orbital matrix `M i j = φ_j(x_i) = exp(-ζ_j ‖x_i - R_j‖²)` for one spin
sector: `x` electron positions, `R` orbital (nuclear) centers, `ζ` orbital exponents. -/
def gaussianOrbitalMatrix {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n, 3]⟩,
    R : ⟨.float, [n, 3]⟩,
    ζ : ⟨.float, [n]⟩
  begin
    let u := Xla.broadcast [⟨n, true⟩, ⟨n, false⟩, ⟨3, true⟩] x;
    let v := Xla.broadcast [⟨n, false⟩, ⟨n, true⟩, ⟨3, true⟩] R;
    let d := Xla.sub u v;
    let d2 := Xla.mul d d;
    let t := Xla.transpose d2 perm[2, 0, 1];
    let r2 := Xla.sum 1 t;
    let w := Xla.broadcast [⟨n, false⟩, ⟨n, true⟩] ζ;
    return Xla.exp (Xla.neg (Xla.mul w r2))

/-- The simple ansatz `Ψ = det M₁ · det M₂`, one Gaussian Slater determinant per spin
sector, with independent orbital parameters (unrestricted form; the restricted ansatz
is the special case `R₁ = R₂`, `ζ₁ = ζ₂`). -/
def slaterDetProd {n₁ n₂ : ℕ} :=
  ssa Xla.XlaOp with
    x₁ : ⟨.float, [n₁, 3]⟩,
    x₂ : ⟨.float, [n₂, 3]⟩,
    R₁ : ⟨.float, [n₁, 3]⟩,
    ζ₁ : ⟨.float, [n₁]⟩,
    R₂ : ⟨.float, [n₂, 3]⟩,
    ζ₂ : ⟨.float, [n₂]⟩
  begin
    let m₁ := gaussianOrbitalMatrix.apply (x₁.append (R₁.append ζ₁));
    let m₂ := gaussianOrbitalMatrix.apply (x₂.append (R₂.append ζ₂));
    return Xla.mul (Xla.det m₁) (Xla.det m₂)

#eval IO.println (slaterDetProd (n₁ := 2) (n₂ := 2)).code

/-! ## Mathematical semantics and correctness statements -/

/-- Mathematical specification of one determinant factor:
`det [exp(-ζ_j ‖x_i - R_j‖²)]`. -/
noncomputable def gaussianDetDef {n : ℕ} (x R : Fin n → Fin 3 → ℝ) (ζ : Fin n → ℝ) : ℝ :=
  Matrix.det fun i j ↦ Real.exp (-(ζ j * ∑ k, (x i k - R j k) ^ 2))

/-- Mathematical specification of the simple ansatz. -/
noncomputable def slaterDetProdDef {n₁ n₂ : ℕ}
    (x₁ : Fin n₁ → Fin 3 → ℝ) (x₂ : Fin n₂ → Fin 3 → ℝ)
    (R₁ : Fin n₁ → Fin 3 → ℝ) (ζ₁ : Fin n₁ → ℝ)
    (R₂ : Fin n₂ → Fin 3 → ℝ) (ζ₂ : Fin n₂ → ℝ) : ℝ :=
  gaussianDetDef x₁ R₁ ζ₁ * gaussianDetDef x₂ R₂ ζ₂

/-- Bridge theorem: the generated XLA code computes exactly `slaterDetProdDef`. -/
theorem slaterDetProd_eq_def {n₁ n₂ : ℕ} :
    Xla.simpleEval (slaterDetProd (n₁ := n₁) (n₂ := n₂)) = slaterDetProdDef := by
  sorry

/-! ### Single-sector lemmas (all assume positive exponents: orbitals must decay) -/

theorem gaussianDet_perm {n : ℕ} (σ : Equiv.Perm (Fin n))
    (x R : Fin n → Fin 3 → ℝ) (ζ : Fin n → ℝ) :
    gaussianDetDef (x ∘ σ) R ζ = σ.sign • gaussianDetDef x R ζ := by
  sorry

/-- In fact C^∞; the spec only needs C². -/
theorem gaussianDet_smooth {n : ℕ} (R : Fin n → Fin 3 → ℝ) (ζ : Fin n → ℝ) :
    ContDiff ℝ ⊤ (fun x ↦ gaussianDetDef x R ζ) := by
  sorry

theorem gaussianDet_sq_integrable {n : ℕ} (R : Fin n → Fin 3 → ℝ) (ζ : Fin n → ℝ)
    (hζ : ∀ j, 0 < ζ j) :
    MeasureTheory.Integrable (fun x ↦ (gaussianDetDef x R ζ) ^ 2) := by
  sorry

theorem gaussianDet_bounded {n : ℕ} (x R : Fin n → Fin 3 → ℝ) (ζ : Fin n → ℝ) :
    |gaussianDetDef x R ζ| ≤ (Nat.factorial n : ℝ) := by
  sorry

theorem gaussianDet_vanishes {n : ℕ} (R : Fin n → Fin 3 → ℝ) (ζ : Fin n → ℝ)
    (hζ : ∀ j, 0 < ζ j) :
    Filter.Tendsto (fun x ↦ gaussianDetDef x R ζ) (Filter.cocompact _) (nhds 0) := by
  sorry

/-! ### The simple ansatz is a valid QMC wavefunction -/

theorem slaterDetProd_valid {n₁ n₂ : ℕ}
    (R₁ : Fin n₁ → Fin 3 → ℝ) (ζ₁ : Fin n₁ → ℝ)
    (R₂ : Fin n₂ → Fin 3 → ℝ) (ζ₂ : Fin n₂ → ℝ)
    (hζ₁ : ∀ j, 0 < ζ₁ j) (hζ₂ : ∀ j, 0 < ζ₂ j) :
    IsValidQMCWavefunction (fun x₁ x₂ ↦ slaterDetProdDef x₁ x₂ R₁ ζ₁ R₂ ζ₂) := by
  sorry

/-! ### Sampling corollaries -/

open MeasureTheory in
/-- With non-degenerate orbitals the normalization constant is positive. -/
theorem slaterDetProd_integral_pos {n₁ n₂ : ℕ}
    (R₁ : Fin n₁ → Fin 3 → ℝ) (ζ₁ : Fin n₁ → ℝ)
    (R₂ : Fin n₂ → Fin 3 → ℝ) (ζ₂ : Fin n₂ → ℝ)
    (hζ₁ : ∀ j, 0 < ζ₁ j) (hζ₂ : ∀ j, 0 < ζ₂ j)
    (h0 : ∃ x₁ x₂, slaterDetProdDef x₁ x₂ R₁ ζ₁ R₂ ζ₂ ≠ 0) :
    0 < ∫ x : (Fin n₁ → Fin 3 → ℝ) × (Fin n₂ → Fin 3 → ℝ),
      (slaterDetProdDef x.1 x.2 R₁ ζ₁ R₂ ζ₂) ^ 2 := by
  sorry

open MeasureTheory in
/-- `|Ψ|² / ∫|Ψ|²` is a probability density — exactly what the Metropolis sampler
(`pfnet/qmc.py:make_sampler`) targets. -/
theorem slaterDetProd_normalized_density {n₁ n₂ : ℕ}
    (R₁ : Fin n₁ → Fin 3 → ℝ) (ζ₁ : Fin n₁ → ℝ)
    (R₂ : Fin n₂ → Fin 3 → ℝ) (ζ₂ : Fin n₂ → ℝ)
    (hζ₁ : ∀ j, 0 < ζ₁ j) (hζ₂ : ∀ j, 0 < ζ₂ j)
    (h0 : ∃ x₁ x₂, slaterDetProdDef x₁ x₂ R₁ ζ₁ R₂ ζ₂ ≠ 0) :
    ∫ x : (Fin n₁ → Fin 3 → ℝ) × (Fin n₂ → Fin 3 → ℝ),
      (slaterDetProdDef x.1 x.2 R₁ ζ₁ R₂ ζ₂) ^ 2 /
        (∫ y : (Fin n₁ → Fin 3 → ℝ) × (Fin n₂ → Fin 3 → ℝ),
          (slaterDetProdDef y.1 y.2 R₁ ζ₁ R₂ ζ₂) ^ 2) = 1 := by
  sorry
