import JaxProof

def fn {n : ℕ} :=
  xla with
    x : float [n],
    m : int []
  returns
    float [n]
  begin
    let loop_fn :=
      xla with
        i : int [],
        x : float [n]
      returns
        float [n]
      begin
        return .bind .mul *[x, x];
    fori_loop loop_fn, m, (.cons x .nil), .nil

#eval IO.println (fn (n := 10)).pretty_print

example (m n : ℕ) (x : Fin n → ℝ) :
    (fn (n := n).eval Jax.FloatAsReal) *[x, (m : ℤ)] = *[x ^ (2 ^ m)] := by
  simp only [fn, Fin.isValue, Jax.ExprGroup.eval, List.cons_append, List.nil_append, List.append_eq]
  induction m with
  | zero =>
    simp [Jax.Expr.eval]
    rfl
  | succ m ih =>
    conv_lhs at ih =>
      arg 3
      change Int.natAbs ↑m
    conv_lhs =>
      arg 3
      change Int.natAbs ↑m + 1
    conv_lhs at ih =>
      arg 1
      change *[x]
    conv_lhs =>
      arg 1
      change *[x]
    simp only [ih]
    simp only [Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval, Int.natAbs_natCast,
      Jax.Tensor.map₂, Jax.DList.cons.injEq, and_true]
    rw [pow_add, pow_one, pow_mul, pow_two]
    rfl

