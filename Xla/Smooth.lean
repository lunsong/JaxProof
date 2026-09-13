import Xla.Impl
import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.Analysis.Calculus.ContDiff.Operations
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# `Xla/Smooth.lean` — smoothness of the DSL semantics

`QMC/Ansatz.lean` demands that every candidate wavefunction be `C²` in the electron
coordinates (`ContDiff ℝ 2`), because `pfnet` evaluates the local energy by automatic
differentiation and silently computes the wrong Laplacian at a crease or a jump (see
the module docstring there). This file is the *leaf library* of that obligation: the
facts about the `DirectImpl` semantics of the individual primitives from which the
smoothness of a whole program is composed along its dataflow.

## The smooth variable

Every lemma is stated for an arbitrary normed `ℝ`-vector space `X` — in a
wavefunction, `X` is the electron-position space — and tensor-valued functions
`f : X → Xla.Tensor ℝ s`. `Tensor ℝ s` is `Curry Fin s ℝ`, a finitely nested `Π` type
of `ℝ`s. `Curry` is an abbreviation whose body matches on the shape list, so typeclass
search cannot unfold that match for a variable `s`; the recursive instances
`Tensor.instNormedAddCommGroup`/`Tensor.instNormedSpace` below therefore build the
canonical product structure on *every* `Tensor ℝ s`, and the lemmas need no per-shape
instance arguments.

Data that a program treats as *constant* in the smooth variable is passed as an
ordinary argument, never as a function of `X`. In particular the integer tensors
(nuclear charges, `gather` indices) never occur under a `C²` hypothesis: `ℤ` carries
no `NormedSpace ℝ` structure, and the DSL only ever uses it for
`iota`/`ofNat`/`mod`/`gather` indices, which are constant in the electron coordinates
(the op discipline of `QMC/Ansatz.lean`).

## Shape-level results

`Tensor.map`/`Tensor.map₂` and the `Tensor.flatten`-based side conditions
(`0 < (f x).flatten i`) are how a shape-generic statement refers to "every entry" of a
tensor without naming its index type.

## The proofs

The leaf statements below are `sorry`; each is a small analysis fact whose proof is
independent of the program it is used in. Program-level smoothness
(`QMC/FermiNet.lean`) is then a composition of these leaves along the dataflow.

Where the work is:

* `contDiff_tensorMap`/`contDiff_tensorMap₂`: induction on the shape `s` — at `[]` the
  statement is `hg.comp` and at `s₀ :: s` it is `contDiff_pi'` plus the induction
  hypothesis. The wrinkle is that the induction hypothesis is stated for the *tail*
  shape, whose normed-space instances are not recoverable from the hypothesis for the
  whole shape (a `Π`-instance does not expose its fibres); proving it for a variable `s`
  therefore wants either the instances threaded through the induction or a
  `Curry`-level reformulation.
* `contDiff_tensorMap_sqrt`/`contDiff_tensorMap₂_div`: pointwise
  (`ContDiffAt` from `Real.contDiffAt_sqrt`/`contDiffAt_div`, using
  `flatten`-componentwise positivity/non-vanishing), then `contDiff_iff_contDiffAt`.
* `contDiff_gather`: the `unflatten ∘ flatten` round trip of a reindexing; the
  `unflatten`/`flatten` isometries are `contDiff_unflatten` plus linearity.
* `contDiff_einsum`/`contDiff_det`/`contDiff_sumN`/`contDiff_transpose`/
  `contDiff_broadcast`/`contDiff_unflatten`: `Tensor.einsum`, `Matrix.det`, `sumN` are
  finite sums of products of entries and the index manipulations are linear isometries;
  `contDiff_pi'` reduces each to the entrywise polynomial/identity.
-/

namespace Xla

open Soir

variable {X : Type} [NormedAddCommGroup X] [NormedSpace ℝ X]

/-! ### Canonical normed structure on tensors

`Tensor ℝ s` is `Curry Fin s ℝ`, whose body matches on `s`; typeclass search cannot
unfold that match for a variable `s`, so the shape-generic statements cannot leave the
normed-space instances to be found. The recursive instances below equip every
`Tensor ℝ s` with the canonical (product) structure, which removes the per-shape
instance arguments entirely. -/

noncomputable instance Tensor.instNormedAddCommGroup :
    (s : Shape) → NormedAddCommGroup (Tensor ℝ s)
  | [] => inferInstanceAs (NormedAddCommGroup ℝ)
  | _ :: s =>
    letI : NormedAddCommGroup (Tensor ℝ s) := Tensor.instNormedAddCommGroup s
    inferInstanceAs (NormedAddCommGroup (Fin _ → Tensor ℝ s))

noncomputable instance Tensor.instNormedSpace :
    (s : Shape) → NormedSpace ℝ (Tensor ℝ s)
  | [] => inferInstanceAs (NormedSpace ℝ ℝ)
  | _ :: s =>
    letI : NormedAddCommGroup (Tensor ℝ s) := Tensor.instNormedAddCommGroup s
    letI : NormedSpace ℝ (Tensor ℝ s) := Tensor.instNormedSpace s
    inferInstanceAs (NormedSpace ℝ (Fin _ → Tensor ℝ s))

/-- `Tensor.map` is `Curry.map`, definitionally after a case split on the shape. -/
private theorem Tensor.map_eq_curryMap (s : Shape) (g : ℝ → ℝ) :
    (Tensor.map g : Tensor ℝ s → Tensor ℝ s) = Curry.map g := by
  induction s with
  | nil => rfl
  | cons s₀ s ih => rfl

/-! ### Elementwise maps -/

/-- `Tensor.map` of a `C²` scalar function preserves `C²`. -/
theorem contDiff_tensorMap {s : Shape} {g : ℝ → ℝ} (hg : ContDiff ℝ 2 g)
    {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (f x).map g := by
  induction s with
  | nil =>
    change ContDiff ℝ 2 fun x => g (f x)
    exact hg.comp hf
  | cons s₀ s ih =>
    change ContDiff ℝ 2 fun x (i : Fin s₀) => Curry.map g (f x i)
    apply contDiff_pi'
    intro i
    rw [← Tensor.map_eq_curryMap s g]
    exact ih ((contDiff_apply ℝ _ i).comp hf)

/-- `Tensor.map₂` of a `C²` scalar function of two variables preserves `C²`. -/
theorem contDiff_tensorMap₂ {s : Shape} {g : ℝ → ℝ → ℝ} (hg : ContDiff ℝ 2 (Function.uncurry g))
    {f₁ f₂ : X → Tensor ℝ s} (h₁ : ContDiff ℝ 2 f₁) (h₂ : ContDiff ℝ 2 f₂) :
    ContDiff ℝ 2 fun x => Tensor.map₂ g (f₁ x) (f₂ x) := by
  induction s with
  | nil =>
    change ContDiff ℝ 2 fun x => g (f₁ x) (f₂ x)
    exact hg.comp (h₁.prodMk h₂)
  | cons s₀ s ih =>
    change ContDiff ℝ 2 fun x (i : Fin s₀) => Tensor.map₂ g (f₁ x i) (f₂ x i)
    apply contDiff_pi'
    intro i
    exact ih ((contDiff_apply ℝ _ i).comp h₁) ((contDiff_apply ℝ _ i).comp h₂)

section Elementwise

variable {s : Shape}

/-- `DirectImpl.exp`. -/
theorem contDiff_tensorMap_exp {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (f x).map Real.exp :=
  contDiff_tensorMap Real.contDiff_exp hf

/-- `DirectImpl.neg`. -/
theorem contDiff_tensorMap_neg {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (f x).map fun t => -t :=
  contDiff_tensorMap (by fun_prop) hf

/-- `DirectImpl.add`. -/
theorem contDiff_tensorMap₂_add {f₁ f₂ : X → Tensor ℝ s} (h₁ : ContDiff ℝ 2 f₁)
    (h₂ : ContDiff ℝ 2 f₂) :
    ContDiff ℝ 2 fun x => Tensor.map₂ (· + ·) (f₁ x) (f₂ x) :=
  contDiff_tensorMap₂ (by fun_prop) h₁ h₂

/-- `DirectImpl.sub`. -/
theorem contDiff_tensorMap₂_sub {f₁ f₂ : X → Tensor ℝ s} (h₁ : ContDiff ℝ 2 f₁)
    (h₂ : ContDiff ℝ 2 f₂) :
    ContDiff ℝ 2 fun x => Tensor.map₂ (· - ·) (f₁ x) (f₂ x) :=
  contDiff_tensorMap₂ (by fun_prop) h₁ h₂

/-- `DirectImpl.mul`. -/
theorem contDiff_tensorMap₂_mul {f₁ f₂ : X → Tensor ℝ s} (h₁ : ContDiff ℝ 2 f₁)
    (h₂ : ContDiff ℝ 2 f₂) :
    ContDiff ℝ 2 fun x => Tensor.map₂ (· * ·) (f₁ x) (f₂ x) :=
  contDiff_tensorMap₂ (by fun_prop) h₁ h₂

/-- Reading a cons-shaped tensor at `i.mulAdd j` is reading its `i`-th component at `j`.
`Tensor.flatten` is defined through `Fin.divNat`/`Fin.modNat`, which rewriting cannot see
through because `List.prod` is semireducible; routing the identity through the
`Tensor.unflatten` round trip keeps it syntactically clean. -/
private theorem Tensor.flatten_mulAdd {R : Type} {s₀ : ℕ} {s : Shape}
    (x : Tensor R (s₀ :: s)) (i : Fin s₀) (j : Fin s.prod) :
    Tensor.flatten x (i.mulAdd j) = Tensor.flatten (x i) j := by
  have h : x i = Tensor.unflatten s (fun j => Tensor.flatten x (i.mulAdd j)) := by
    rw [← congrFun (Tensor.unflatten_flatten x) i]
    rfl
  rw [h]
  exact (congrFun (Tensor.flatten_unflatten s (fun j => Tensor.flatten x (i.mulAdd j))) j).symm

/-- `DirectImpl.div`: `C²` where the denominator does not vanish. (In the DSL,
division only ever appears inside `tanh = 1 - 2/(exp(2x)+1)`, whose denominator is
`≥ 1`.) -/
theorem contDiff_tensorMap₂_div {f₁ f₂ : X → Tensor ℝ s} (h₁ : ContDiff ℝ 2 f₁)
    (h₂ : ContDiff ℝ 2 f₂) (hne : ∀ x (i : Fin s.prod), (f₂ x).flatten i ≠ 0) :
    ContDiff ℝ 2 fun x => Tensor.map₂ (· / ·) (f₁ x) (f₂ x) := by
  rw [contDiff_iff_contDiffAt]
  intro x₀
  induction s with
  | nil =>
    change ContDiffAt ℝ 2 (fun x => f₁ x / f₂ x) x₀
    exact h₁.contDiffAt.div h₂.contDiffAt (hne x₀ ⟨0, by simp⟩)
  | cons s₀ s ih =>
    change ContDiffAt ℝ 2 (fun x (i : Fin s₀) => Tensor.map₂ (· / ·) (f₁ x i) (f₂ x i)) x₀
    apply contDiffAt_pi'
    intro i
    exact ih ((contDiff_apply ℝ _ i).comp h₁) ((contDiff_apply ℝ _ i).comp h₂)
      (fun x j => by
        have h := hne x (i.mulAdd j)
        rw [Tensor.flatten_mulAdd (x := f₂ x) i j] at h
        exact h)

/-- `DirectImpl.sqrt`: `C²` where the argument is positive. The DSL guards every
`sqrt` by `√(r² + ε)` with `ε = exp θ > 0` (see `enDist`/`pairDist`), so the argument
is bounded away from the non-analytic point `0`. -/
theorem contDiff_tensorMap_sqrt {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f)
    (hpos : ∀ x (i : Fin s.prod), 0 < (f x).flatten i) :
    ContDiff ℝ 2 fun x => (f x).map Real.sqrt := by
  rw [contDiff_iff_contDiffAt]
  intro x₀
  induction s with
  | nil =>
    change ContDiffAt ℝ 2 (fun x => Real.sqrt (f x)) x₀
    exact (Real.contDiffAt_sqrt (ne_of_gt (hpos x₀ ⟨0, by simp⟩))).comp x₀ hf.contDiffAt
  | cons s₀ s ih =>
    change ContDiffAt ℝ 2 (fun x (i : Fin s₀) => (f x i).map Real.sqrt) x₀
    apply contDiffAt_pi'
    intro i
    rw [← Tensor.map_eq_curryMap s Real.sqrt]
    exact ih ((contDiff_apply ℝ _ i).comp hf)
      (fun x j => by
        have h := hpos x (i.mulAdd j)
        rw [Tensor.flatten_mulAdd (x := f x) i j] at h
        exact h)

end Elementwise

/-! ### Reductions and index manipulation -/

/-- A finite sum of tensors commutes with the evaluation of a `Curry` argument:
addition on `Tensor` is pointwise, but `Finset.sum` does not unfold definitionally,
so this is an induction on the finset. -/
private theorem Tensor.sum_apply_get {ι : Type} {s₀ : ℕ} {s : Shape}
    (t : Finset ι) (x : ι → Tensor ℝ (s₀ :: s)) (i : Fin s₀) :
    (∑ j ∈ t, x j) i = ∑ j ∈ t, x j i := by
  classical
  induction t using Finset.induction_on with
  | empty => simp
  | insert a t ha ih =>
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    show x a i + (∑ j ∈ t, x j) i = x a i + ∑ j ∈ t, x j i
    rw [ih]

/-- A finite sum of `C²` tensor-valued functions is `C²`, proved entrywise. -/
private theorem contDiff_finset_sum {ι : Type} [Fintype ι]
    {s : Shape} {g : ι → X → Tensor ℝ s} (hg : ∀ i, ContDiff ℝ 2 (g i)) :
    ContDiff ℝ 2 fun x => ∑ i, g i x := by
  induction s with
  | nil =>
    with_unfolding_all
      exact ContDiff.sum (𝕜 := ℝ) (n := 2) (s := Finset.univ)
        (f := fun i x => g i x) (fun i _ => hg i)
  | cons s₀ s ih =>
    change ContDiff ℝ 2 fun x (j : Fin s₀) => (∑ i, g i x) j
    apply contDiff_pi'
    intro j
    have hfun : (fun x => (∑ i, g i x) j) = fun x => ∑ i, g i x j := by
      funext x
      rw [Tensor.sum_apply_get Finset.univ]
    rw [hfun]
    exact ih (fun i => (contDiff_apply ℝ _ j).comp (hg i))

/-- `DirectImpl.sum`: a finite sum of `C²` components. -/
theorem contDiff_sumN {s : Shape} (n : ℕ) {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (f x).sumN n := by
  induction s generalizing n with
  | nil =>
    cases n with
    | zero =>
      change ContDiff ℝ 2 fun x => f x
      exact hf
    | succ n =>
      change ContDiff ℝ 2 fun x => f x
      exact hf
  | cons s₀ s ih =>
    cases n with
    | zero =>
      change ContDiff ℝ 2 fun x => f x
      exact hf
    | succ n =>
      change ContDiff ℝ 2 fun x => ((f x).sumFirst).sumN n
      refine ih n ?_
      change ContDiff ℝ 2 fun x => ∑ i : Fin s₀, f x i
      exact contDiff_finset_sum (fun i => (contDiff_apply ℝ _ i).comp hf)

/-- `DirectImpl.einsum` of two input tensors: a finite sum of products of entries. -/
theorem contDiff_einsum (s : Shape) (i₁ i₂ : List (Fin s.length)) (n : ℕ)
    [NormedAddCommGroup (Tensor ℝ (i₁.map s.get))]
    [NormedSpace ℝ (Tensor ℝ (i₁.map s.get))]
    [NormedAddCommGroup (Tensor ℝ (i₂.map s.get))]
    [NormedSpace ℝ (Tensor ℝ (i₂.map s.get))]
    [NormedAddCommGroup (Tensor ℝ (s.drop n))] [NormedSpace ℝ (Tensor ℝ (s.drop n))]
    {f₁ : X → Tensor ℝ (i₁.map s.get)} {f₂ : X → Tensor ℝ (i₂.map s.get)}
    (h₁ : ContDiff ℝ 2 f₁) (h₂ : ContDiff ℝ 2 f₂) :
    ContDiff ℝ 2 fun x => Tensor.einsum s [⟨i₁, f₁ x⟩, ⟨i₂, f₂ x⟩] n := by
  sorry

/-- `DirectImpl.det`: a polynomial (in fact a sum of products of the entries, via
`Matrix.det`). -/
theorem contDiff_det {n : ℕ} {f : X → Tensor ℝ [n, n]} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => Matrix.det (f x) := by
  have h : (fun x => Matrix.det (f x)) = fun x => ∑ σ : Equiv.Perm (Fin n),
      (Equiv.Perm.sign σ : ℝ) * ∏ i, f x (σ i) i := by
    funext x
    exact Matrix.det_apply' (M := (f x : Matrix (Fin n) (Fin n) ℝ))
  rw [h]
  fun_prop

/-- `DirectImpl.transpose`: a permutation of the indices is a linear isometry. -/
theorem contDiff_transpose {s : Shape} (σ : Equiv.Perm (Fin s.length))
    [NormedAddCommGroup (Tensor ℝ s)] [NormedSpace ℝ (Tensor ℝ s)]
    [NormedAddCommGroup (Tensor ℝ (List.ofFn fun i => s.get (σ i)))]
    [NormedSpace ℝ (Tensor ℝ (List.ofFn fun i => s.get (σ i)))]
    {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (f x).transpose σ := by
  sorry

/-- `DirectImpl.broadcast`: duplicating entries along new axes is a linear isometry. -/
theorem contDiff_broadcast {s : List (ℕ × Bool)}
    [NormedAddCommGroup (Tensor ℝ (Tensor.preBroadcast s))]
    [NormedSpace ℝ (Tensor ℝ (Tensor.preBroadcast s))]
    [NormedAddCommGroup (Tensor ℝ (s.map Prod.fst))]
    [NormedSpace ℝ (Tensor ℝ (s.map Prod.fst))]
    {f : X → Tensor ℝ (Tensor.preBroadcast s)} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => Tensor.broadcast s (f x) := by
  sorry

/-- `DirectImpl.unflatten`: reinterpreting a flat index as a multi-index is a linear
isometry. -/
theorem contDiff_unflatten (s : Shape) {f : X → Tensor ℝ [s.prod]} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (Tensor.unflatten s (f x) : Tensor ℝ s) := by
  induction s with
  | nil =>
    change ContDiff ℝ 2 fun x => Tensor.unflatten [] (f x)
    simp only [Tensor.unflatten]
    exact (contDiff_apply ℝ ℝ 0).comp hf
  | cons n s ih =>
    change ContDiff ℝ 2 fun x (i : Fin n) => Tensor.unflatten s (fun j => f x (i.mulAdd j))
    apply contDiff_pi'
    intro i
    apply ih
    apply contDiff_pi'
    intro j
    exact (contDiff_apply ℝ ℝ (i.mulAdd j)).comp hf

/-- `DirectImpl.gather` from a one-dimensional table along a *fixed* integer index
tensor: a reindexing of the data, hence `C²` in the data. The index is constant in the
smooth variable by the op discipline; this is the only `gather` shape the FermiNet
program uses (parameter slicing and the nuclear-charge embedding).

Entrywise, `(gather x idx).flatten r = x (Fin.intCast (idx.flatten r))` holds
definitionally (`simp [DirectImpl.gather, reduce_tensor, reduce_soir, reduce_xla]`),
so the statement below is `gather` up to the `unflatten ∘ flatten` round trip
(`Tensor.unflatten_flatten`). -/
theorem contDiff_gather {n : ℕ} [NeZero n] {s' : Shape} (idx : Tensor ℤ s')
    {f : X → Tensor ℝ [n]} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x =>
      Tensor.unflatten s' fun r => f x (Fin.intCast (idx.flatten r)) := by
  apply contDiff_unflatten s'
  apply contDiff_pi'
  intro r
  exact (contDiff_apply ℝ ℝ (Fin.intCast (idx.flatten r))).comp hf

end Xla
