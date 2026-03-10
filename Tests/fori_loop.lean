import JaxProof

xla_fun sqr (n : ℕ)
arguments
  i : int [],
  x : float [n]
returns
  float [n]
begin
  return .binop .mul x x

def fn (m n : ℕ) : Jax.ExprGroup [⟨.float, [n]⟩] [⟨.float, [n]⟩] :=
  .fori_loop m (sqr n) (.cons (.arg 0) .nil) .nil

#eval IO.println (fn 2 3).code

example (m n : ℕ) (x : Fin n → ℝ) :
    ((fn m n).eval Jax.FloatAsReal) ⟨x, ()⟩ = ⟨x ^ (2 ^ m), ()⟩ := by
  simp [Jax.ExprGroup.eval, Jax.FloatAsReal, fn]
  induction m with
  | zero =>
    simp [Jax.Expr.eval, Jax.Tuple.get]
    rfl
  | succ m ih =>
    simp [ih]
    simp [sqr, Jax.TensorImpl.ofNat, Jax.Tuple.append, Jax.Exprs.toExprGroup,
      Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl]
    conv_lhs =>
      arg 1; arg 2
      change x ^ 2 ^ m
    conv_lhs =>
      arg 1; arg 3
      change x ^ 2 ^ m
    rw [pow_succ, pow_mul, pow_two]
    rfl
