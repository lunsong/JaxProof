import Xla

def matmul {n m l : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n, m]⟩, ⟨.float, [m, l]⟩]
    ⟨.float, [n, l]⟩ :=
  Soir.Expr.ofFn fun x y =>
    let x := Xla.transpose x [0, 1].formPerm;
    Xla.dot_general [] [m] [n] [l] x y

#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code

example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval x y = x * y := by
  ext i j
  simp [matmul, reduce_xla, reduce_soir, reduce_tensor]
  congr
  erw [Finset.sum_apply, Finset.sum_apply]
  rfl
