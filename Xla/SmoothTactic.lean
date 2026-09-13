import Xla.Smooth
import Lean

/-!
# `Xla/SmoothTactic.lean` — the `contDiff_eval` tactic

`Xla/Smooth.lean` proves the smoothness of the individual DSL primitives. Composing
those leaves into the smoothness of a whole program is a purely syntactic recursion
along the dataflow: at every node, the goal is `ContDiff ℝ 2 fun x => e` and the body
`e` has a head operation (a primitive, a library call, or one of the component
functions that compose them), so exactly one leaf or node lemma applies.

This file provides the tactic that performs that recursion. It is a small,
head-directed variant of `continuity`/`fun_prop`: unlike those it does not search the
whole library but only the lemmas tagged `@[contDiff_eval_rule]`, and it dispatches on
the *head* of the goal's body expression, so each node costs one `apply` instead of
one per candidate rule. Rules are collected from the environment at tactic runtime,
so tagging a new lemma is all that is needed to extend the tactic.

Rules whose conclusion is `ContDiff ℝ 2 fun x => (lib).eval (f x)` are indexed by the
head constant of `lib` (e.g. `FermiNet.paramSlice`); rules whose conclusion is
`ContDiff ℝ 2 fun x => Tensor.map₂ (· + ·) (f₁ x) (f₂ x)` are indexed by the head of
the body (`Tensor.map₂`). The body of the goal after unfolding a program definition is
in evaluated form (`Expr.eval DirectImpl lib ⋯ 0`); the library `lib` is extracted from
the `Expr.eval` application, and the rule is *specialized* to it before `apply`, which
avoids the dependent-type unification failures of `apply` on the bare rule.

Subgoals whose target is not a `ContDiff` goal — the positivity/non-vanishing side
conditions of `sqrt`/`div`, for instance — are not recursed into; they are returned
as remaining goals so the caller can discharge them with `positivity` or by hand.
-/

open Lean Meta Elab Tactic

namespace Xla

variable {X : Type} [NormedAddCommGroup X] [NormedSpace ℝ X]

/-- Pointwise evaluation of a `C²` product-valued function at a fixed index: the
projection out of the `Curry` product. It is the shape of the `vmap` semantics, where
the batched function is evaluated at the batch index. -/
@[contDiff_eval_rule]
theorem contDiff_eval_apply {ι : Type} [Fintype ι] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] {f : X → (ι → E)}
    (hf : ContDiff ℝ 2 f) (i : ι) :
    ContDiff ℝ 2 fun x => f x i :=
  (contDiff_apply ℝ E i).comp hf

/-- Coordinatewise smoothness of a `Tensor`-valued function: the product decomposition
dual to `contDiff_eval_apply`. The codomain is fixed to `Tensor ℝ s` so that instance
inference uses the canonical `Tensor` norm instead of trying to normalize the (often
unreduced) codomain type of a `vmap`ed goal. -/
@[contDiff_eval_rule]
theorem contDiff_eval_curry_pi {ι : Type} [Fintype ι] {s : Shape}
    {f : X → ι → Tensor ℝ s}
    (h : ∀ i, ContDiff ℝ 2 fun x => f x i) : ContDiff ℝ 2 f :=
  contDiff_pi.mpr h

/-- If `e` is an application of `SimpleExpr.eval`/`Expr.eval`, return the library
argument that is being evaluated. The argument position is fixed by the signature of
`Expr.eval` (`impl` explicit, `Impl` instance, `expr`), so no type inference is
needed, which keeps this usable under binders. -/
private def evalLibExpr? (e : Expr) : Option Expr :=
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``Xla.SimpleExpr.eval then
    if args.size > 2 then some args[2]! else none
  else if fn.isConstOf ``Soir.Expr.eval then
    if args.size > 6 then some args[6]! else none
  else
    none

/-- Simplify an expression with the `reduce_soir` dsimprocs (the definitional part of
`simp only [reduce_soir]`), to consume the `Index`/`Curry` scaffolding that evaluation
leaves around library calls. -/
private def reduceSoirBody (e : Expr) : MetaM Expr := do
  let ctx ← Simp.mkContext
  let simprocs : Simp.SimprocsArray :=
    #[← Lean.Meta.Simp.SimprocExtension.getSimprocs ReduceSoirProc]
  let (e', _) ← Meta.dsimp e ctx simprocs
  return e'

/-- Reduce an `id a b ⋯` application to `a b ⋯`. Normalization uses this instead of
`whnf`, which would unfold the library evaluation under the `id` as well. -/
private def stripId (e : Expr) : Expr :=
  let args := e.getAppArgs
  if 0 < args.size then mkAppN args[0]! (args.eraseIdx! 0) else e

/-- `Index.cons`/`Index.append`/... heads that the DSL leaves in evaluated terms when a
rule's `apply` instantiates a function argument to an `Index` application at a literal
position (`fun x => Index.cons (f x) (Index.cons ...) ⟨1, _⟩` instead of `fun x => f x`).
They reduce definitionally to the selected component. -/
private def isIndexHead (fn : Expr) : Bool :=
  fn.isConstOf ``Soir.Index.cons || fn.isConstOf ``Soir.Index.append ||
  fn.isConstOf ``Soir.Index.single || fn.isConstOf ``Soir.Index.map ||
  fn.isConstOf ``Soir.Index.unmap || fn.isConstOf ``Soir.Index.replicate

/-- Consume the `id` and `Index` scaffolding that evaluation leaves around library
calls (see `isIndexHead`), at most `fuel` layers deep. -/
private partial def normalizeBody (body : Expr) (fuel : Nat := 16) : MetaM Expr := do
  if fuel == 0 then return body
  let fn := body.getAppFn
  if fn.isConstOf ``id then
    normalizeBody (stripId body) (fuel - 1)
  else if isIndexHead fn then
    let body' ← reduceSoirBody body
    if body'.equal body then return body else normalizeBody body' (fuel - 1)
  else
    return body

/-- The discriminator of a goal or rule body and, for `.eval` bodies, the library
expression. `Index` heads left behind by unification are reduced first (definitionally)
so that the underlying operation is exposed. -/
private def bodyInfo? (body : Expr) : MetaM (Option (Name × Option Expr)) := do
  let body ← normalizeBody body
  let fn := body.getAppFn
  if fn.isConstOf ``Xla.SimpleExpr.eval || fn.isConstOf ``Soir.Expr.eval then
    let some lib := evalLibExpr? body | return none
    return some (lib.getAppFn.constName!, some lib)
  else if fn.isConst then
    return some (fn.constName!, none)
  else
    return none

/-- The function argument `f` of a `ContDiff 𝕜 n f` target. -/
private def contDiffFun? (ty : Expr) : Option Expr := do
  guard (ty.getAppFn.isConstOf ``ContDiff)
  return ty.getAppArgs.back!

/-- The info of the body of the lambda `f`. The lambda variable is introduced into the
local context (via `lambdaTelescope`) so that the body can be reduced; the result is a
name and (for `.eval` bodies) a library expression, neither of which mentions the
lambda variable. -/
private def lambdaInfo? (f : Expr) : MetaM (Option (Name × Option Expr)) := do
  let f ← instantiateMVars f
  if f.isLambda then
    lambdaTelescope f fun _ body => bodyInfo? body
  else
    return none

/-- The info of a rule: discriminator and library of the body of its conclusion. -/
private def ruleInfo? (r : Name) : MetaM (Option (Name × Option Expr)) :=
  withoutModifyingState do
    let info ← getConstInfo r
    forallTelescopeReducing info.type fun _ concl => do
      let some f := contDiffFun? concl | return none
      lambdaInfo? f

/-- The info of a goal. -/
private def goalInfo? (g : MVarId) : MetaM (Option (Name × Option Expr)) :=
  withoutModifyingState do
    let ty ← g.getType
    let some f := contDiffFun? ty | return none
    lambdaInfo? f

/-- Normalize the goal's function body, consuming the `Index` scaffolding that `apply`
of a rule (in particular `Xla.contDiff_einsum`) leaves behind (see `bodyInfo?`). The
target is replaced by a definitionally equal one (`MVarId.replaceTargetDefEq`), so the
rules below see the underlying operation directly. Returns the goal to continue with. -/
private def normalizeContDiffGoal (g : MVarId) : MetaM MVarId := do
  let ty ← g.getType
  let some f := contDiffFun? ty | return g
  let f ← instantiateMVars f
  if !f.isLambda then return g
  let some f' ← lambdaTelescope f fun xs body => do
    let body' ← normalizeBody body
    if body'.equal body then return none
    return some (← mkLambdaFVars xs body')
    | return g
  let args := ty.getAppArgs
  let ty' := mkAppN ty.getAppFn (args.set! (args.size - 1) f')
  g.replaceTargetDefEq ty'

/-- Specialize an `.eval` rule to the goal's library before applying it: assign the
rule's leading arguments by unifying its library subterm with `goalLib`. This keeps
`apply` from having to unify the whole dependent type of the conclusion at once (which
fails for the `einsum` input types); the result behaves like
`apply (rule args…)` with the arguments extracted from the goal. -/
private def specializeEvalRule (r : Name) (goalLib : Expr) : MetaM (Option Expr) := do
  let e ← mkConstWithFreshMVarLevels r
  let eType ← inferType e
  let (mvarIds, _, concl) ← forallMetaTelescopeReducing eType
  let some f := contDiffFun? concl | return none
  let some e' ← lambdaTelescope f fun _ body => do
    let some ruleLib := evalLibExpr? body | return none
    if ← withTransparency .default (isDefEq ruleLib goalLib) then
      return some (← instantiateMVars (mkAppN e mvarIds))
    else
      return none
    | return none
  return some e'

/-- Rules that are tried on any goal without a more specific discriminator: constant
functions, projections, and pointwise evaluation at a fixed index (the `vmap`
projection). -/
private def fallbackRules : Array Name :=
  #[``Xla.contDiff_eval_apply, ``contDiff_const, ``contDiff_fst, ``contDiff_snd]

/-- All tagged rules, grouped by discriminator. -/
private def collectRules : MetaM (Std.HashMap Name (Array Name)) := do
  let env ← getEnv
  let mut map : Std.HashMap Name (Array Name) := {}
  for r in contDiffEvalRulesExt.getState env do
    if let some (d, _) ← ruleInfo? r then
      map := map.insert d (map.getD d #[] |>.push r)
  return map

/-- Try applying rule `r` to `g` at default transparency, specializing `.eval` rules to
the goal's library expression when there is one. -/
private def tryApply (r : Name) (goalLib? : Option Expr) (g : MVarId) :
    TacticM (Option (List MVarId)) := do
  let e ← match goalLib? with
    | some lib =>
      match ← specializeEvalRule r lib with
      | some e' => pure e'
      | none => mkConstWithFreshMVarLevels r
    | none => mkConstWithFreshMVarLevels r
  try
    return some (← withTransparency .default (g.apply e { approx := false }))
  catch _ =>
    return none

/-- Close `g` from a local hypothesis. -/
private def tryAssumption (g : MVarId) : TacticM Bool := do
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let s ← saveState
    try
      let sub ← withTransparency .default (g.apply d.toExpr { approx := false })
      if sub.isEmpty then return true
      restoreState s
    catch _ =>
      restoreState s
  return false

/-- A goal is solvable by the recursion if it is a `ContDiff` goal or a `∀` whose body
is (after `intro`) a solvable goal; other `∀`s (side conditions) are left pending by
`solveGoal`. -/
private def isSolvableGoal (g : MVarId) : TacticM Bool := do
  let ty ← g.getType
  return ty.getAppFn.isConstOf ``ContDiff || ty.isForall

/-- The goal's function is a lambda returning a lambda: the codomain is a product, and
`contDiff_eval_curry_pi` decomposes the goal coordinatewise. -/
private def goalIsNestedLambda (g : MVarId) : MetaM Bool := withoutModifyingState do
  let ty ← g.getType
  let some f := contDiffFun? ty | return false
  let f ← instantiateMVars f
  if !f.isLambda then return false
  return f.bindingBody!.isLambda

/--
Recursively solve the `ContDiff` goal `g`. Returns the non-`ContDiff` subgoals left
over (side conditions), or `none` if some `ContDiff` subgoal could not be closed.
-/
private partial def solveGoal (rules : Std.HashMap Name (Array Name)) (g : MVarId) :
    TacticM (Option (Array MVarId)) := do
  let ty ← g.getType
  if ty.isForall then
    let s ← saveState
    try
      let (_, g') ← g.intro1P
      if ← isSolvableGoal g' then
        match ← solveGoal rules g' with
        | some pending =>
          if pending.isEmpty then return some #[]
          -- leave the whole `∀` as a side condition rather than one with the
          -- introduced variables fixed
          restoreState s
          return some #[g]
        | none => restoreState s; return none
      else
        restoreState s
        return some #[g]
    catch _ => restoreState s; return none
  let g ← normalizeContDiffGoal g
  if ← tryAssumption g then return some #[]
  let info? ← goalInfo? g
  let (d?, lib?) := match info? with
    | some (d, lib) => (some d, lib)
    | none => (none, none)
  let nested ← goalIsNestedLambda g
  let piIntro : Array Name := if nested then #[``Xla.contDiff_eval_curry_pi] else #[]
  let candidates := match d? with
    | some d => piIntro ++ rules.getD d #[] ++ fallbackRules
    | none => piIntro ++ fallbackRules
  for r in candidates do
    let s ← saveState
    match ← tryApply r lib? g with
    | none => restoreState s
    | some subs =>
      let (cd, other) ← subs.partitionM isSolvableGoal
      let mut pending : Array MVarId := other.toArray
      let mut ok := true
      for sg in cd do
        match ← solveGoal rules sg with
        | some p => pending := pending ++ p
        | none => ok := false; break
      if ok then return some pending
      restoreState s
  return none

/--
`contDiff_eval` proves goals of the form `ContDiff ℝ 2 fun x => e` by recursively
applying the `@[contDiff_eval_rule]` lemmas along the head structure of `e`. It leaves
non-`ContDiff` side conditions (positivity of `sqrt` arguments, non-vanishing
denominators) as remaining goals.
-/
elab "contDiff_eval" : tactic => do
  let rules ← collectRules
  let g ← getMainGoal
  match ← solveGoal rules g with
  | some pending => replaceMainGoal pending.toList
  | none => throwError "contDiff_eval failed to prove the goal"

end Xla
