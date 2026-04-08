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

#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).pretty_print
/-
%0: transpose [1, 0] $0
%1: dot_general 1 0 %0 $1
return %1, 
-/

example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp only [matmul, List.length_cons, List.length_nil, Nat.reduceAdd, List.formPerm_cons_cons,
    List.formPerm_singleton, Jax.ExprGroup.eval, Jax.DList.cons.injEq, and_true]
  apply funext
  intro i
  apply funext
  intro j
  conv_lhs =>
    change (∑ k, fun i j => x i k * y k j) i j
  simp [Matrix.mul_apply, Finset.sum_apply]

  
