import Lean

open Lean
open Lean.Meta
open Lean.Meta.Simp

initialize ReduceXLA : SimpExtension ← registerSimpAttr `reduce_xla "unfold xla evaluation"

initialize ReduceTensor : SimpExtension ← registerSimpAttr `reduce_tensor "unfold tensor evaluation"

/-- Simprocs for `reduce_tensor`: the paired simproc attribute `reduce_tensor_proc` is
looked up when `@[reduce_tensor]` is applied to a simproc, see `Soir/Meta.lean`. -/
initialize ReduceTensorProc : SimprocExtension ←
  registerSimprocAttr `reduce_tensor_proc "simprocs for tensor evaluation" none

/-- Rules for the `contDiff_eval` tactic (`Xla/SmoothTactic.lean`), accumulated across
modules: `SimplePersistentEnvExtension` folds the imported states with `addImportedFn`,
so rules tagged in `Xla/Smooth.lean` are visible to proofs in downstream modules. -/
initialize contDiffEvalRulesExt : SimplePersistentEnvExtension Name (Array Name) ←
  registerSimplePersistentEnvExtension {
    name := `Xla.contDiffEvalRules
    addImportedFn := fun ess => ess.foldl (init := #[]) (· ++ ·)
    addEntryFn := fun es r => es.push r
  }

/-- `@[contDiff_eval_rule]` marks a smoothness lemma (conclusion `ContDiff ℝ 2 f`) as a
rule for the `contDiff_eval` tactic. Unlike `TagAttribute`, this attribute can tag
declarations of imported modules (the rule registry above is a plain environment
extension without the current-module restriction). -/
initialize registerBuiltinAttribute {
  name := `contDiff_eval_rule
  descr := "lemmas applied by the `contDiff_eval` tactic"
  applicationTime := .afterCompilation
  add := fun decl _stx _kind =>
    modifyEnv fun env => contDiffEvalRulesExt.addEntry env decl
  erase := fun _ => throwError "cannot remove `contDiff_eval_rule`"
}
