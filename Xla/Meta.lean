import Lean

open Lean Meta in
initialize ReduceXLA : SimpExtension ← registerSimpAttr `reduce_xla "unfold xla evaluation"

open Lean Meta in
initialize ReduceTensor : SimpExtension ← registerSimpAttr `reduce_tensor "unfold tensor evaluation"
