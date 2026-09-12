import Xla

open Xla Soir

def transposeSwap {n m : ℕ} : Xla.SimpleExpr [⟨.float, [n, m]⟩] ⟨.float, [m, n]⟩ :=
  Soir.Expr.ofFn fun x => Xla.transpose x perm[1, 0]

def sumTransposed {n l : ℕ} :
    Xla.SimpleExpr [⟨.float, [n, 3, l]⟩] ⟨.float, [l, n]⟩ :=
  Soir.Expr.ofFn fun x => Xla.sum 1 (Xla.transpose x perm[1, 2, 0])

/-- The index casts `Tensor.transpose` inserts are erased. -/
example {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (i : Fin m) (j : Fin n) :
    transposeSwap.eval x i j = x j i := by
  simp [transposeSwap, reduce_xla, reduce_soir, reduce_tensor]

/-- `Tensor.sumN` unfolds even though the shape of the transposed tensor is a
`List.ofFn` rather than a syntactic cons tower. -/
example {n l : ℕ} (x : Xla.Tensor ℝ [n, 3, l]) (i : Fin l) (k : Fin n) :
    sumTransposed.eval x i k = ∑ j : Fin 3, x k j i := by
  simp [sumTransposed, reduce_xla, reduce_soir, reduce_tensor]
  rfl
