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

Each leaf statement below is a small analysis fact whose proof is
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

/-- `Curry.get` at a fixed (multi-)index is the corresponding product projection, so it
preserves `C²`. -/
private theorem contDiff_curryGet : ∀ {s : Shape} (i : Index Fin s) {f : X → Tensor ℝ s},
    ContDiff ℝ 2 f → ContDiff ℝ 2 fun x => (f x).get i := by
  intro s
  induction s with
  | nil =>
    intro i f hf
    change ContDiff ℝ 2 f
    exact hf
  | cons s₀ s ih =>
    intro i f hf
    change ContDiff ℝ 2 fun x => ((f x) (i ⟨0, by simp⟩)).get (fun r => i r.succ)
    exact ih (fun r => i r.succ) ((contDiff_apply ℝ _ (i ⟨0, by simp⟩)).comp hf)

/-- `Curry.of` rebuilds a tensor from its index function, entrywise. -/
private theorem contDiff_curryOf : ∀ {s : Shape} {g : X → (Index Fin s → ℝ)},
    ContDiff ℝ 2 g → ContDiff ℝ 2 fun x => (Curry.of (g x) : Tensor ℝ s) := by
  intro s
  induction s with
  | nil =>
    intro g hg
    change ContDiff ℝ 2 fun x => g x Index.null
    exact (contDiff_apply ℝ ℝ Index.null).comp hg
  | cons s₀ s ih =>
    intro g hg
    change ContDiff ℝ 2 fun x (v : Fin s₀) =>
      (Curry.of (fun a => g x (Index.cons v a)) : Tensor ℝ s)
    apply contDiff_pi'
    intro v
    apply ih
    apply contDiff_pi'
    intro a
    exact (contDiff_apply ℝ ℝ (Index.cons v a)).comp hg

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

/-! ### Entrywise formula for `Tensor.einprod` -/

/-- `Index.select` along a label list is the corresponding `cons` of projections. -/
private theorem Index.select_cons {ι : Type} {m : ι → Type} {γ : List ι}
    (a : Fin γ.length) (is : List (Fin γ.length)) (w : Index m γ) :
    Index.select (a :: is) w = Index.cons (w a) (Index.select is w) := by
  funext r
  match r with
  | ⟨0, _⟩ => rfl
  | ⟨k + 1, _⟩ => rfl

/-- An index in cons form is determined by its head and tail. -/
private theorem Index.cons_eta {ι : Type} {m : ι → Type} {γ₀ : ι} {γ : List ι}
    (v : Index m (γ₀ :: γ)) : Index.cons (v 0) (fun r => v r.succ) = v := by
  funext r
  match r with
  | ⟨0, _⟩ => rfl
  | ⟨k + 1, _⟩ => rfl

/-- `Curry.get` at a cons index decomposes as the head projection and the tail `get`. -/
private theorem Curry.get_cons {ι : Type} {m : ι → Type} {α : Type} {γ₀ : ι} {γ : List ι}
    (f : Curry m (γ₀ :: γ) α) (a : m γ₀) (b : Index m γ) :
    f.get (Index.cons a b) = (f a).get b := rfl

private theorem filter_pred_cons_zero {n : ℕ} (i : List (Fin (n + 1))) :
    filter_pred ((0 : Fin (n + 1)) :: i) = filter_pred i := rfl

private theorem filter_pred_cons_succ {n : ℕ} (j : Fin n) (i : List (Fin (n + 1))) :
    filter_pred (j.succ :: i) = j :: filter_pred i := rfl

/-- `Tensor.einprod.filter` contracts the axes labeled `0` at the fixed value `i₀`;
reading the result at the remaining labels is reading the original tensor at the
corresponding multi-index. -/
private theorem Tensor.einprod.filter_get {R : Type} [Mul R] [One R]
    {s₀ : ℕ} {s' : List ℕ} (i₀ : Fin s₀) (v' : Index Fin s')
    (i : List (Fin (s'.length + 1))) (x : Tensor R (i.map (s₀ :: s').get)) :
    (Tensor.einprod.filter s₀ s' i₀ i x).get
        (Index.select (γ := s') (filter_pred i) v') =
      x.get (Index.select (γ := s₀ :: s') i (Index.cons i₀ v')) := by
  revert x
  induction i with
  | nil =>
    intro x
    rfl
  | cons a is ih =>
    intro x
    induction a using Fin.cases with
    | zero =>
      change (Tensor.einprod.filter s₀ s' i₀ is (x i₀)).get
        (Index.select (γ := s') (filter_pred is) v') =
        x.get (Index.select (γ := s₀ :: s') (0 :: is) (Index.cons i₀ v'))
      rw [Index.select_cons, Curry.get_cons, Index.cons_zero]
      exact ih (x i₀)
    | succ j =>
      change (Tensor.einprod.filter s₀ s' i₀ is (x (v' j))).get
        (Index.select (γ := s') (filter_pred is) v') =
        x.get (Index.select (γ := s₀ :: s') (j.succ :: is) (Index.cons i₀ v'))
      rw [Index.select_cons, Curry.get_cons, Index.cons_succ]
      exact ih (x (v' j))

/-- The entries of `Tensor.einprod` are the products of the entries selected by the
label lists; an induction on the shape. -/
private theorem Tensor.einprod_get {R : Type} [Mul R] [One R] : ∀ (s : List ℕ)
    (xs : List ((i : List (Fin s.length)) × Tensor R (i.map s.get)))
    (v : Index Fin s),
    (Tensor.einprod s xs).get v =
      (xs.map fun p => p.2.get (Index.select (γ := s) p.1 v)).prod
  | [], xs, v => by
    change (xs.map fun ⟨i, x⟩ => match i with | [] => x).prod =
      (xs.map fun p => p.2.get (Index.select (γ := []) p.1 v)).prod
    congr 1
    apply List.map_congr_left
    rintro ⟨i, x⟩ -
    cases i with
    | nil => rfl
    | cons h _ => exact Fin.elim0 h
  | s₀ :: s', xs, v => by
    obtain ⟨i₀, v', rfl⟩ : ∃ (i₀ : Fin s₀) (v' : Index Fin s'), v = Index.cons i₀ v' :=
      ⟨v 0, fun r => v r.succ, (Index.cons_eta v).symm⟩
    change (Tensor.einprod s'
        (xs.map fun p => ⟨filter_pred p.1,
          Tensor.einprod.filter s₀ s' i₀ p.1 p.2⟩)).get v' =
      (xs.map fun p => p.2.get
        (Index.select (γ := s₀ :: s') p.1 (Index.cons i₀ v'))).prod
    rw [Tensor.einprod_get s'
      (xs.map fun p => ⟨filter_pred p.1, Tensor.einprod.filter s₀ s' i₀ p.1 p.2⟩)
      v', List.map_map]
    congr 1
    apply List.map_congr_left
    intro p _
    change (Tensor.einprod.filter s₀ s' i₀ p.1 p.2).get
      (Index.select (γ := s') (filter_pred p.1) v') =
      p.2.get (Index.select (γ := s₀ :: s') p.1 (Index.cons i₀ v'))
    rw [Tensor.einprod.filter_get]

/-- `DirectImpl.einsum` of two input tensors: a finite sum of products of entries. -/
theorem contDiff_einsum (s : Shape) (i₁ i₂ : List (Fin s.length)) (n : ℕ)
    {f₁ : X → Tensor ℝ (i₁.map s.get)} {f₂ : X → Tensor ℝ (i₂.map s.get)}
    (h₁ : ContDiff ℝ 2 f₁) (h₂ : ContDiff ℝ 2 f₂) :
    ContDiff ℝ 2 fun x => Tensor.einsum s [⟨i₁, f₁ x⟩, ⟨i₂, f₂ x⟩] n := by
  apply contDiff_sumN
  have h : (fun x => Tensor.einprod s [⟨i₁, f₁ x⟩, ⟨i₂, f₂ x⟩]) =
      fun x => Curry.of fun v =>
        (f₁ x).get (Index.select (γ := s) i₁ v) *
          (f₂ x).get (Index.select (γ := s) i₂ v) := by
    funext x
    rw [← Curry.of_get (Tensor.einprod s [⟨i₁, f₁ x⟩, ⟨i₂, f₂ x⟩])]
    congr 1
    funext v
    rw [Tensor.einprod_get]
    simp
  rw [h]
  apply contDiff_curryOf
  apply contDiff_pi'
  intro v
  exact (contDiff_curryGet (Index.select (γ := s) i₁ v) h₁).mul
    (contDiff_curryGet (Index.select (γ := s) i₂ v) h₂)

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
    {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (f x).transpose σ := by
  apply contDiff_curryOf
  apply contDiff_pi'
  intro i
  exact contDiff_curryGet _ hf

/-- `DirectImpl.broadcast`: duplicating entries along new axes is a linear isometry. -/
theorem contDiff_broadcast {s : List (ℕ × Bool)}
    {f : X → Tensor ℝ (Tensor.preBroadcast s)} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => Tensor.broadcast s (f x) := by
  induction s with
  | nil => exact hf
  | cons hd tl ih =>
    obtain ⟨a, b⟩ := hd
    cases b with
    | true =>
      change ContDiff ℝ 2 fun x (i : Fin a) => Tensor.broadcast tl (f x i)
      apply contDiff_pi'
      intro i
      exact ih ((contDiff_apply ℝ _ i).comp hf)
    | false =>
      change ContDiff ℝ 2 fun x (_ : Fin a) => Tensor.broadcast tl (f x)
      apply contDiff_pi'
      intro _
      exact ih hf

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
