import JaxProof

namespace Example

def norm_xla_verion {n : ℕ} :=
  xla with
    x : float [n]
  returns
    float []
  begin
    let_expr x2 : float [n] := .bind .mul *[x, x];
    let_expr x2_sumed : float [] := .bind (.sum 1) *[x2];
    return .bind .sqrt *[x2_sumed]

def normalize_xla_verion {n : ℕ} :=
  let f₀ := Jax.ExprGroup.cons (Jax.Expr.arg 0) (norm_xla_verion (n := n))
  let f₁ :=
    xla with
      x : float [n],
      norm_x : float []
    returns
      float [n]
    begin
      let_expr norm_x : float [n] := .bind (.broadcast [(n, false)]) *[norm_x];
      return .bind .div *[x, norm_x]
  Jax.ExprGroup.apply f₀ f₁

#eval IO.println (normalize_xla_verion (n := 12)).code

noncomputable def norm {n : ℕ} (x : Fin n → ℝ) : ℝ := √(∑ i, (x i)^2)

noncomputable def normalize {n : ℕ} (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  x i / norm x

theorem norm_def (n : ℕ) (x : Fin n → ℝ) :
    norm_xla_verion.eval Jax.FloatAsReal *[x] = *[norm x] := by
  simp only [norm_xla_verion, Fin.isValue, Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl,
    Jax.Expr.eval.recursive_eval, Jax.Tensor.map₂, Jax.Tensor.sumN, Jax.Tensor.sumFirst,
    List.tail_cons, Jax.Tensor.map, norm, Jax.DList.cons.injEq, and_true]
  congr
  ext i
  rw [pow_two]
  rfl

theorem normalize_def (n : ℕ) (x : Fin n → ℝ) : 
    normalize_xla_verion.eval Jax.FloatAsReal *[x] = *[normalize x] := by
  simp only [normalize_xla_verion, List.length_cons, List.length_nil, Nat.reduceAdd, Fin.isValue,
    Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval,
    Jax.Tensor.map₂, Jax.DList.cons.injEq, and_true]
  apply funext
  intro i
  conv_lhs =>
    arg 1
    change x i
  conv_lhs =>
    arg 2
    change Jax.DList.get 0 (norm_xla_verion.eval Jax.FloatAsReal *[x])
    rw [norm_def]
    change norm x
  rfl

end Example
