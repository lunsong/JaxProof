import Lean

open Lean Meta in
initialize ReduceXLA : SimpExtension ← registerSimpAttr `reduce_xla "unfold xla evaluation"

open Lean Meta in
initialize ReduceTensor : SimpExtension ← registerSimpAttr `reduce_tensor "unfold tensor evaluation"

open Lean Meta Simp in
/-- Simprocs for `reduce_tensor`: the paired simproc attribute `reduce_tensor_proc` is
looked up when `@[reduce_tensor]` is applied to a simproc, see `Soir/Meta.lean`. -/
initialize ReduceTensorProc : SimprocExtension ←
  registerSimprocAttr `reduce_tensor_proc "simprocs for tensor evaluation" none
