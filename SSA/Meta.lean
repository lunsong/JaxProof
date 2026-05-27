import Lean

open Lean Meta in
initialize ReduceSSA : SimpExtension ← registerSimpAttr `reduce_ssa "unfold ssa evaluation"
