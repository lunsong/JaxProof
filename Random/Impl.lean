import Soir.Core
import Random.Op
import Random.Meta
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Independence.InfinitePi
import Mathlib.Probability.Independence.Integration

namespace Random

open Soir
open MeasureTheory
open ProbabilityTheory

inductive RandVarName where
  | normal : ℕ → RandVarName
  | uniform : ℕ → RandVarName
deriving DecidableEq

/-- The law of a single draw: `normal k` is a standard normal variable and
`uniform k` is uniform on `[0, 1]`. -/
noncomputable def RandVarName.law : RandVarName → Measure ℝ
  | .normal _ => gaussianReal 0 1
  | .uniform _ => volume.restrict (Set.Icc 0 1)

instance : IsProbabilityMeasure (volume.restrict (Set.Icc (0 : ℝ) 1)) :=
  ⟨by rw [Measure.restrict_apply_univ, Real.volume_Icc]; norm_num⟩

instance (name : RandVarName) : IsProbabilityMeasure (RandVarName.law name) := by
  cases name with
  | normal _ => exact (inferInstance : IsProbabilityMeasure (gaussianReal 0 1))
  | uniform _ =>
    exact (inferInstance : IsProbabilityMeasure (volume.restrict (Set.Icc (0 : ℝ) 1)))

/-- The joint law of all draws: they are independent and the law of each one is
`RandVarName.law`. -/
noncomputable def law : Measure (RandVarName → ℝ) :=
  Measure.infinitePi RandVarName.law

instance : IsProbabilityMeasure law := by
  rw [law]; infer_instance

/-- A random variable is a real number depending on the draws. -/
abbrev RandVar := (RandVarName → ℝ) → ℝ

/-- The mean (expectation) of a random variable: its average over all draws. -/
noncomputable def RandVar.mean (x : RandVar) : ℝ :=
  ∫ ω, x ω ∂law

namespace RandVar

@[simp] lemma mean_neg (x : RandVar) : mean (-x) = -mean x := by
  simpa only [mean, Pi.neg_apply] using integral_neg x

@[simp] lemma mean_add (x y : RandVar)
    (hx : Integrable x law) (hy : Integrable y law) :
    mean (x + y) = mean x + mean y := by
  simpa only [mean, Pi.add_apply] using integral_add hx hy

@[simp] lemma mean_sub (x y : RandVar)
    (hx : Integrable x law) (hy : Integrable y law) :
    mean (x - y) = mean x - mean y := by
  simpa only [mean, Pi.sub_apply] using integral_sub hx hy

@[simp] lemma mean_const_mul (c : ℝ) (x : RandVar) :
    mean ((fun _ : RandVarName → ℝ => c) * x) = c * mean x := by
  simpa only [mean, Pi.mul_apply] using integral_const_mul c x

@[simp] lemma mean_mul_const (c : ℝ) (x : RandVar) :
    mean (x * (fun _ : RandVarName → ℝ => c)) = mean x * c := by
  simpa only [mean, Pi.mul_apply] using integral_mul_const c x

@[simp] lemma mean_div_const (c : ℝ) (x : RandVar) :
    mean (x / (fun _ : RandVarName → ℝ => c)) = mean x / c := by
  simpa only [mean, Pi.div_apply] using integral_div c x

lemma mean_mul_of_indep (x y : RandVar)
    (hxy : IndepFun x y law)
    (hx : AEStronglyMeasurable x law) (hy : AEStronglyMeasurable y law) :
    mean (x * y) = mean x * mean y := by
  simpa only [mean, Pi.mul_apply] using hxy.integral_mul_eq_mul_integral hx hy

/-- The `name`-th draw is distributed according to `RandVarName.law name`. -/
lemma measurePreserving_eval_law (name : RandVarName) :
    MeasurePreserving (fun ω : RandVarName → ℝ => ω name) law (RandVarName.law name) := by
  simpa [law] using measurePreserving_eval_infinitePi RandVarName.law name

/-- Distinct draws are independent. -/
lemma indepFun_eval_law {i j : RandVarName} (hij : i ≠ j) :
    IndepFun (fun ω : RandVarName → ℝ => ω i) (fun ω => ω j) law := by
  have h := iIndepFun_infinitePi (P := RandVarName.law) (X := fun _ (x : ℝ) => x)
    fun _ => measurable_id
  simpa [law] using h.indepFun hij

@[simp] lemma integrable_eval_normal (k : ℕ) :
    Integrable (fun ω : RandVarName → ℝ => ω (.normal k)) law := by
  have hg : Integrable (fun x : ℝ => x) (gaussianReal 0 1) := by
    rw [← memLp_one_iff_integrable]
    exact memLp_id_gaussianReal 1
  simpa [Function.comp_def, RandVarName.law] using
    (measurePreserving_eval_law (.normal k)).integrable_comp_of_integrable
      (g := fun x : ℝ => x) hg

@[simp] lemma integrable_eval_uniform (k : ℕ) :
    Integrable (fun ω : RandVarName → ℝ => ω (.uniform k)) law := by
  have hg : Integrable (fun x : ℝ => x) (volume.restrict (Set.Icc (0 : ℝ) 1)) :=
    (intervalIntegrable_iff_integrableOn_Icc_of_le (by norm_num : (0 : ℝ) ≤ 1)).mp
      (continuous_id.intervalIntegrable 0 1)
  simpa [Function.comp_def, RandVarName.law] using
    (measurePreserving_eval_law (.uniform k)).integrable_comp_of_integrable
      (g := fun x : ℝ => x) hg

@[simp] lemma integrable_sq_eval_normal (k : ℕ) :
    Integrable ((fun ω : RandVarName → ℝ => ω (.normal k)) * (fun ω => ω (.normal k))) law := by
  have hg : Integrable (fun x : ℝ => x * x) (gaussianReal 0 1) := by
    simpa only [id_eq, pow_two] using MemLp.integrable_sq (memLp_id_gaussianReal 2)
  change Integrable ((fun x : ℝ => x * x) ∘ fun ω : RandVarName → ℝ => ω (.normal k)) law
  simpa [Function.comp_def, RandVarName.law] using
    (measurePreserving_eval_law (.normal k)).integrable_comp_of_integrable
      (g := fun x : ℝ => x * x) hg

@[simp] lemma integrable_sq_eval_uniform (k : ℕ) :
    Integrable ((fun ω : RandVarName → ℝ => ω (.uniform k)) * (fun ω => ω (.uniform k))) law := by
  have hg : Integrable (fun x : ℝ => x * x) (volume.restrict (Set.Icc (0 : ℝ) 1)) :=
    show IntegrableOn (fun x : ℝ => x * x) (Set.Icc (0 : ℝ) 1) volume from
      (intervalIntegrable_iff_integrableOn_Icc_of_le (f := fun x : ℝ => x * x)
        (by norm_num : (0 : ℝ) ≤ 1)).mp
        ((continuous_id.mul continuous_id).intervalIntegrable (μ := volume) 0 1)
  change Integrable ((fun x : ℝ => x * x) ∘ fun ω : RandVarName → ℝ => ω (.uniform k)) law
  simpa [Function.comp_def, RandVarName.law] using
    (measurePreserving_eval_law (.uniform k)).integrable_comp_of_integrable
      (g := fun x : ℝ => x * x) hg

@[simp] lemma integrable_const_mul {x : RandVar} (hx : Integrable x law) (c : ℝ) :
    Integrable ((fun _ : RandVarName → ℝ => c) * x) law := by
  change Integrable (fun ω : RandVarName → ℝ => c * x ω) law
  exact hx.const_mul c

/-- The mean of `f` applied to a single draw. -/
lemma mean_eval_comp (name : RandVarName) {f : ℝ → ℝ} (hf : Measurable f) :
    mean (fun ω : RandVarName → ℝ => f (ω name)) = ∫ x, f x ∂RandVarName.law name := by
  simp only [mean]
  rw [← integral_map (measurable_pi_apply name).aemeasurable (hf.aestronglyMeasurable)]
  rw [(measurePreserving_eval_law name).map_eq]

/-- The mean of a single draw. -/
lemma mean_eval (name : RandVarName) :
    mean (fun ω : RandVarName → ℝ => ω name) = ∫ x, x ∂RandVarName.law name :=
  mean_eval_comp name measurable_id

@[simp] lemma mean_normal (k : ℕ) :
    mean (fun ω : RandVarName → ℝ => ω (.normal k)) = 0 := by
  simpa [RandVarName.law] using mean_eval (.normal k)

@[simp] lemma mean_uniform (k : ℕ) :
    mean (fun ω : RandVarName → ℝ => ω (.uniform k)) = 1 / 2 := by
  rw [mean_eval, RandVarName.law]
  rw [integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le (by norm_num : (0 : ℝ) ≤ 1)]
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt
    (f := fun y : ℝ => y ^ 2 / 2) (f' := fun y => y)]
  · norm_num
  · intro x _
    simpa using (hasDerivAt_pow 2 x).div_const 2
  · exact continuous_id.intervalIntegrable 0 1

@[simp] lemma mean_sq_normal (k : ℕ) :
    mean ((fun ω : RandVarName → ℝ => ω (.normal k)) * (fun ω => ω (.normal k))) = 1 := by
  change mean (fun ω : RandVarName → ℝ => (fun x : ℝ => x * x) (ω (.normal k))) = 1
  rw [mean_eval_comp (.normal k) (f := fun x : ℝ => x * x) (by fun_prop), RandVarName.law]
  have h := variance_fun_id_gaussianReal (μ := 0) (v := 1)
  rw [variance_eq_sub (X := fun x : ℝ => x) (memLp_id_gaussianReal 2),
    integral_id_gaussianReal] at h
  norm_num at h
  simpa only [pow_two] using h

@[simp] lemma mean_sq_uniform (k : ℕ) :
    mean ((fun ω : RandVarName → ℝ => ω (.uniform k)) * (fun ω => ω (.uniform k))) = 1 / 3 := by
  change mean (fun ω : RandVarName → ℝ => (fun x : ℝ => x * x) (ω (.uniform k))) = 1 / 3
  rw [mean_eval_comp (.uniform k) (f := fun x : ℝ => x * x) (by fun_prop), RandVarName.law]
  rw [integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le (by norm_num : (0 : ℝ) ≤ 1)]
  simp only [← pow_two]
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt
    (f := fun y : ℝ => y ^ 3 / 3) (f' := fun y => y ^ 2)]
  · norm_num
  · intro x _
    simpa using (hasDerivAt_pow 3 x).div_const 3
  · exact (continuous_id.pow 2).intervalIntegrable 0 1

@[simp] lemma mean_mul_normal_of_ne {k l : ℕ} (h : k ≠ l) :
    mean ((fun ω : RandVarName → ℝ => ω (.normal k)) * (fun ω => ω (.normal l))) = 0 := by
  rw [mean_mul_of_indep _ _ (indepFun_eval_law (by simpa using h))
    (measurable_pi_apply _).aestronglyMeasurable
    (measurable_pi_apply _).aestronglyMeasurable]
  simp only [mean_normal, mul_zero]

end RandVar

abbrev RandType.impl : RandType → Type
  | .data => RandVar
  | .key => ℕ

@[reduce_random]
noncomputable instance RandVarImpl : SimpleImpl RandPrimOp RandType.impl where
  bind op := match op with
  | .ofNat n => fun _ => n
  | .add => fun f g => f + g
  | .sub => fun f g => f - g
  | .mul => fun f g => f * g
  | .div => fun f g => f / g
  | .neg => fun f => -f
  | .shuffle => (· + 1)
  | .normal => fun k => fun x => x (.normal k)
  | .uniform => fun k => fun x => x (.uniform k)

abbrev SimpleExpr (args : List RandType) (out : RandType) : Type :=
  Expr RandOp args [out]

@[reduce_random]
noncomputable def SimpleExpr.eval
  {args : List RandType} {out : RandType}
  (expr : SimpleExpr args out) :
    Curry RandType.impl args (RandType.impl out) :=
  (Expr.eval RandType.impl expr).map fun x => x 0

/-!
## Draws at a tail of the key space

An expression evaluated at key `m` only reads draws at keys `≥ m`, because
`shuffle` is the only operation acting on keys and it adds one.  Its value is
therefore measurable with respect to the σ-algebra generated by the draws at
keys `≥ m`, hence independent of any single draw at a key `< m`.
-/

-- Unfold the type-level evaluation helpers during `simp` and `dsimp`; they are
-- transparent in the proofs below but not part of the public interface.
set_option allowUnsafeReducibility true in
attribute [local reducible] Soir.evalType Soir.Impl.bindType Soir.evalType.bind

/-- The numeric key addressed by a draw. -/
def RandVarName.key : RandVarName → ℕ
  | .normal n => n
  | .uniform n => n

/-- The σ-algebra generated by the single draw `j`. -/
@[instance_reducible]
def tailGen (j : RandVarName) : MeasurableSpace (RandVarName → ℝ) :=
  (inferInstance : MeasurableSpace ℝ).comap fun ω => ω j

/-- The σ-algebra generated by all draws at keys `≥ m`. -/
@[instance_reducible]
noncomputable def tailAlg (m : ℕ) : MeasurableSpace (RandVarName → ℝ) :=
  ⨆ j : RandVarName, ⨆ (_ : m ≤ j.key), tailGen j

lemma tailAlg_le (m : ℕ) :
    tailAlg m ≤ (inferInstance : MeasurableSpace (RandVarName → ℝ)) := by
  rw [tailAlg]
  exact iSup_le fun j => iSup_le fun _ => (measurable_pi_apply j).comap_le

/-- A value of type `α.impl` is *good* for the tail `≥ m` if key values are `≥ m`
and data is measurable with respect to `tailAlg m`. -/
def RImpl (m : ℕ) : (α : RandType) → RandType.impl α → Prop
  | .key => fun j => m ≤ j
  | .data => fun g => Measurable[tailAlg m] g

/-- An index all of whose components are good for the tail `≥ m`. -/
def RIndex (m : ℕ) {outs : List RandType} (x : Index RandType.impl outs) : Prop :=
  ∀ r, RImpl m outs[r] (x r)

lemma RIndex.append {m : ℕ} {outs outs' : List RandType}
    (u : Index RandType.impl outs) (v : Index RandType.impl outs')
    (hu : RIndex m u) (hv : RIndex m v) :
    RIndex m (Index.append u v) := by
  induction outs with
  | nil =>
    intro r
    simpa [Index.append] using hv r
  | cons a as ih =>
    intro r
    match r with
    | ⟨0, hr⟩ =>
      show RImpl m a (u 0)
      exact hu 0
    | ⟨k + 1, hr⟩ =>
      have hk : k < (as ++ outs').length := by
        rw [List.length_append]
        simp only [List.length_cons, List.length_append] at hr
        omega
      exact ih (fun r => u r.succ) (fun r => hu r.succ) ⟨k, hk⟩

lemma RIndex.select {m : ℕ} {outs : List RandType}
    (is : List (Fin outs.length)) (u : Index RandType.impl outs)
    (hu : RIndex m u) :
    RIndex m (Index.select is u) := by
  induction is with
  | nil =>
    intro r
    exact r.elim0
  | cons a is ih =>
    intro r
    match r with
    | ⟨0, hr⟩ =>
      show RImpl m outs[a] (u a)
      exact hu a
    | ⟨k + 1, hr⟩ =>
      have hk : k < (is.map outs.get).length := by
        rw [List.length_map]
        simp at hr
        omega
      exact ih ⟨k, hk⟩

lemma RIndex.single {m : ℕ} {out : RandType} (v : RandType.impl out)
    (hv : RImpl m out v) :
    RIndex m (Index.single v) := by
  intro r
  fin_cases r
  simpa [Index.single] using hv

set_option maxHeartbeats 400000 in
-- Case-splitting on the nine primitive operations and unfolding their semantics
-- exceeds the default heartbeat limit.
/-- Each primitive operation preserves goodness. -/
lemma RIndex.primImpl (m : ℕ) {ins : List RandType} {out : RandType}
    (p : RandPrimOp ins out) (y : Index RandType.impl ins) (hy : RIndex m y) :
    RImpl m out ((SimpleImpl.bind p).get y) := by
  cases p <;> simp only [RImpl, SimpleImpl.bind, reduce_soir]
  · exact measurable_const
  · exact (hy 0).add (hy 1)
  · exact (hy 0).sub (hy 1)
  · exact (hy 0).mul (hy 1)
  · exact (hy 0).div (hy 1)
  · exact (hy 0).neg
  · exact Nat.le_add_right_of_le (hy 0)
  · rw [measurable_iff_comap_le]
    exact le_iSup₂_of_le (RandVarName.normal (y 0)) (hy 0) le_rfl
  · rw [measurable_iff_comap_le]
    exact le_iSup₂_of_le (RandVarName.uniform (y 0)) (hy 0) le_rfl

/-- Binding a primitive operation preserves goodness. -/
lemma RIndex.bind_prim {m : ℕ} {ins : List RandType} {out : RandType}
    (p : RandPrimOp ins out) (y : Index RandType.impl ins) (hy : RIndex m y) :
    RIndex m ((Impl.bind (SimpleOp.simple p) : Curry RandType.impl ins
      (Index RandType.impl [out])).get y) := by
  change RIndex m (((SimpleImpl.bind p).map Index.single).get y)
  rw [Curry.get_map]
  exact RIndex.single _ (RIndex.primImpl m p y hy)

set_option maxHeartbeats 800000 in
-- The induction goes through every constructor of `Expr` and each case unfolds
-- the `Curry`/`Index` evaluation, which exceeds the default heartbeat limit.
/-- Evaluating any expression at key `m` yields good values: key outputs are
`≥ m` and data outputs are measurable with respect to `tailAlg m`. -/
lemma eval_tail (m : ℕ) : ∀ {args outs : List RandType} (expr : Expr RandOp args outs)
    (i : Index RandType.impl args),
    RIndex m i → RIndex m ((Expr.eval RandType.impl expr).get i) := by
  intro args outs expr
  induction expr with
  | nil => intro i h r; exact r.elim0
  | arg i0 =>
    intro i h r
    fin_cases r
    simp only [Expr.eval, Curry.get_map, Index.single, Curry.get_arg]
    exact h i0
  | append x y ihx ihy =>
    intro i h r
    simp only [Expr.eval, Curry.get_map₂]
    exact RIndex.append _ _ (ihx i h) (ihy i h) r
  | select is x ih =>
    intro i h r
    simp only [Expr.eval, Curry.get_map]
    exact RIndex.select is _ (ih i h) r
  | apply f xs ihf ihxs =>
    intro i h r
    simp only [Expr.eval, Curry.get_map]
    exact ihf _ (ihxs i h) r
  | bind op fs xs ihfs ihxs =>
    intro i h r
    cases op with
    | simple p =>
      simp only [Expr.eval, evalType.bind, Curry.get_map]
      exact RIndex.bind_prim p _ (ihxs i h) r

/-- The evaluation of a `SimpleExpr [.key] .data` at key `m` is measurable with
respect to the σ-algebra generated by the draws at keys `≥ m`. -/
theorem SimpleExpr.measurable_eval_tail (m : ℕ) (f : SimpleExpr [.key] .data) :
    Measurable[tailAlg m] (f.eval m) := by
  have h := eval_tail m f (Index.single m)
    (RIndex.single (out := .key) m (by simp [RImpl]))
  exact h 0

/-- A single draw at key `k` is independent of the σ-algebra generated by the
draws at keys `≥ k + 1`. -/
lemma indep_eval_normal_tail (k : ℕ) :
    Indep (tailGen (RandVarName.normal k)) (tailAlg (k + 1)) law := by
  have h_indep : iIndep tailGen (Measure.infinitePi RandVarName.law) := by
    have h := iIndepFun_infinitePi (P := RandVarName.law)
      (X := fun _ (x : ℝ) => x) fun _ => measurable_id
    exact h.iIndep
  have hdisj : Disjoint ({RandVarName.normal k} : Set RandVarName) {j | k + 1 ≤ j.key} := by
    rw [Set.disjoint_left]
    intro j hj hjk
    simp only [Set.mem_singleton_iff] at hj
    subst hj
    simp [RandVarName.key] at hjk
  have hmain := indep_iSup_of_disjoint (m := tailGen) (fun i => (measurable_pi_apply i).comap_le)
    h_indep hdisj
  simpa [tailAlg, law] using hmain

namespace RandVar

/-- Multiplying a tail-measurable variable by an independent standard normal
draw at key `k` has mean zero. -/
lemma mean_mul_normal_tail (k : ℕ) (g : RandVar)
    (hg : Measurable[tailAlg (k + 1)] g) :
    mean ((fun ω : RandVarName → ℝ => ω (.normal k)) * g) = 0 := by
  have hInd : IndepFun (fun ω : RandVarName → ℝ => ω (.normal k)) g law := by
    rw [IndepFun_iff_Indep]
    exact indep_of_indep_of_le_right (indep_eval_normal_tail k)
      (measurable_iff_comap_le.mp hg)
  rw [mean_mul_of_indep _ _ hInd
    (measurable_pi_apply (RandVarName.normal k)).aestronglyMeasurable
    (hg.mono (tailAlg_le (k + 1)) le_rfl).aestronglyMeasurable]
  simp

end RandVar

end Random
