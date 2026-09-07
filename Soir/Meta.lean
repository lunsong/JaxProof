import Lean

open Lean Meta in
initialize ReduceSoir : SimpExtension ← registerSimpAttr `reduce_soir "unfold soir evaluation"
