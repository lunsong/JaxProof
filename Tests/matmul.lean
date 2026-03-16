import JaxProof

def transpose {args : List Jax.TensorType} {α : Jax.DType} {s : List ℕ}
  (σ : Equiv.Perm (Fin s.length)) (x : Jax.Expr args ⟨α, s⟩) :
    Jax.Expr args ⟨α, List.ofFn fun i => s[σ i]⟩ :=
  .bind (.transpose σ) *[x]

def matmul {n m l : ℕ} :=
  xla with
    x : float [n, m],
    y : float [m, l]
  returns
    float [n, l]
  begin
    let_expr x : float [m, n] := transpose [0,1].formPerm x;
    let_expr z : float [n, l] := .bind (.dot_general [] [m] [n] [l]) *[x, y];
    return z

example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl,
    Jax.Expr.eval.recursive_eval, Jax.Tensor.cast]
  apply funext
  intro i
  apply funext
  intro j
  conv_lhs =>
    change (∑ k, fun i j => x i k * y k j) i j
  simp [Matrix.mul_apply, Finset.sum_apply]

  
