import Lean

open Lean
open Lean.Meta
open Lean.Meta.Simp

initialize ReduceRandom : SimpExtension ← registerSimpAttr `reduce_random "unfold random evaluation"
