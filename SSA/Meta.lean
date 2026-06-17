import Lean

open Lean Meta in
initialize ReduceSSA : SimpExtension ← registerSimpAttr `reduce_ssa "unfold ssa evaluation"

open Lean Meta in
initialize ReduceTensor : SimpExtension ← registerSimpAttr `reduce_tensor "unfold tensor evaluation"
