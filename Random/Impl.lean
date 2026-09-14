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
  depOn : Finset RandVarName
  val : (RandVarName → ℝ) → ℝ

abbrev RandType.impl : RandType → Type
  | .data => RandVar
  | .key => ℕ

@[reduce_random]
noncomputable instance RandVarImpl : SimpleImpl RandPrimOp RandType.impl where
  bind op := match op with
  | .ofNat n => ⟨∅, n⟩
  | .add => fun ⟨S₀, f⟩ ⟨S₁, g⟩ => ⟨S₀ ∪ S₁, f + g⟩
  | .sub => fun ⟨S₀, f⟩ ⟨S₁, g⟩ => ⟨S₀ ∪ S₁, f - g⟩
  | .mul => fun ⟨S₀, f⟩ ⟨S₁, g⟩ => ⟨S₀ ∪ S₁, f * g⟩
  | .div => fun ⟨S₀, f⟩ ⟨S₁, g⟩ => ⟨S₀ ∪ S₁, f / g⟩
  | .neg => fun ⟨S, f⟩ => ⟨S, -f⟩
  | .key => 0
  | .shuffle => (· + 1) 
  | .normal => fun k => ⟨{.normal k}, fun x => x (.normal k)⟩
  | .uniform => fun k => ⟨{.uniform k}, fun x => x (.uniform k)⟩

def SimpleExpr (args : List RandType) (out : RandType) : Type :=
  Expr RandOp args [out]

@[reduce_random]
noncomputable def SimpleExpr.eval
  {args : List RandType} {out : RandType}
  (expr : SimpleExpr args out) :
    Curry RandType.impl args (RandType.impl out) :=
  (Expr.eval RandType.impl expr).map fun x => x 0

end Random
