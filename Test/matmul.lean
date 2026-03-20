import JaxProof

def matmul {n m l : ℕ} :=
  xla with
    x : float [n, m],
    y : float [m, l]
  returns
    float [n, l]
  begin
    let_expr x : float [m, n] := Jax.transpose [0,1].formPerm x;
    let_expr z : float [n, l] := Jax.dot_general [] [m] [n] [l] x y;
    return z

#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code

example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval]
  apply funext
  intro i
  apply funext
  intro j
  conv_lhs =>
    change (∑ k, fun i j => x i k * y k j) i j
  simp [Matrix.mul_apply, Finset.sum_apply]

  
