import JaxProof

def fn (m n : ℕ) :=
  xla with
    x : float [n]
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
    fori_loop m, loop_fn, (.cons x .nil), .nil

#eval IO.println (fn 2 3).code

example (m n : ℕ) (x : Fin n → ℝ) :
    ((fn m n).eval Jax.FloatAsReal) *[x] = *[x ^ (2 ^ m)] := by
  simp only [fn, Fin.isValue, Jax.ExprGroup.eval, List.cons_append, List.nil_append, List.append_eq]
  induction m with
  | zero =>
    simp [Jax.Expr.eval]
    rfl
  | succ m ih =>
    simp only [Fin.isValue, ih, Jax.DList.cons.injEq, and_true]
    simp only [Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval]
    rw [pow_add, pow_one, pow_mul, pow_two]
    rfl

