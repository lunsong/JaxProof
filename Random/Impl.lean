import Soir.Core
import Random.Op
import Random.Meta
import Mathlib.Probability.Distributions.Gaussian.Real

namespace Random

open Soir
open MeasureTheory

/-- Semantics of a `measure` value: the joint law of the independent
standard normal coordinates the value depends on.

`dependentOn` lists the keys (random variables) the value depends on, and
`measure` is the joint law of those coordinates: the `i`-th coordinate of the
measure corresponds to `dependentOn[i]`. -/
structure MeasureImpl where
  dependentOn : Finset ℕ
  measure : Measure (Fin dependentOn.card → ℝ)

/-- The standard normal random variable associated with key `k`. -/
noncomputable def MeasureImpl.normal (k : ℕ) : MeasureImpl where
  dependentOn := {k}
  measure :=
    Measure.map (fun x : ℝ => fun _ : Fin [k].length => x)
      (ProbabilityTheory.gaussianReal 0 1)

/-- Product of two measures: the joint law of the independent coordinates of
both. The dependency lists are concatenated. If the two measures depend on a
common key the corresponding coordinates are treated as independent copies
(the current semantics assumes disjoint dependencies). -/
noncomputable def MeasureImpl.prod (m₁ m₂ : MeasureImpl) : MeasureImpl where
  dependentOn := m₁.dependentOn ∪ m₂.dependentOn
  measure := sorry

abbrev RandType.impl : RandType → Type
  | .data .real => ℝ
  | .data .random => MeasureImpl
  | .key => ℕ

@[reduce_random]
noncomputable instance : SimpleImpl RandPrimOp RandType.impl where
  bind op := sorry

def SimpleExpr (args : List RandType) (out : RandType) : Type :=
  Expr RandOp args [out]

@[reduce_random]
noncomputable def SimpleExpr.eval
  {args : List RandType} {out : RandType}
  (expr : SimpleExpr args out) :
    Curry RandType.impl args (RandType.impl out) :=
  (Expr.eval RandType.impl expr).map fun x => x 0

end Random
