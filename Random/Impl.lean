import Soir.Core
import Random.Op
import Random.Meta
import Mathlib.Probability.Distributions.Gaussian.Real

namespace Random

open Soir
open MeasureTheory

inductive RandVarName where
  | normal : ℕ → RandVarName
  | uniform : ℕ → RandVarName
deriving DecidableEq

/-- Semantics of a `measure` value: the joint law of the independent
standard normal coordinates the value depends on.

`dependentOn` lists the keys (random variables) the value depends on, and
`measure` is the joint law of those coordinates: the `i`-th coordinate of the
measure corresponds to `dependentOn[i]`. -/
structure RandVar where
  dependentOn : Finset RandVarName
  measure : Measure (Fin dependentOn.card → ℝ)

def RandVar.promote (S : Finset RandVarName) (x : RandVar) : RandVar where
  dependentOn := x.dependentOn ∪ S
  measure := sorry

abbrev RandType.impl : RandType → Type
  | .data => RandVar
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
