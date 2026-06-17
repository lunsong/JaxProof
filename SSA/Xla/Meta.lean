import Lean

open Lean Meta in
initialize ReduceXLA : SimpExtension ← registerSimpAttr `reduce_xla "unfold xla evaluation"
