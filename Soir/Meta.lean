import Lean

open Lean Meta Simp

initialize ReduceSoir : SimpExtension ← registerSimpAttr `reduce_soir "unfold soir evaluation"

/-- Simprocs for `reduce_soir`. When `simp [reduce_soir]` is elaborated, the paired
simproc attribute `reduce_soir_proc` is looked up automatically (see
`Lean.Meta.Simp.getSimprocExtension?`), and applying `@[reduce_soir]` to a simproc
declaration delegates to it (see `Lean.Meta.Simp.mkSimpAttr`). -/
initialize ReduceSoirProc : SimprocExtension ←
  registerSimprocAttr `reduce_soir_proc "simprocs for soir evaluation" none
