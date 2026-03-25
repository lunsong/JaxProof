import JaxProof

def indexNonzero {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    let_expr x : int [n] := .bind .choice *[x, 1, 0];
    let_expr x_id : int [n] := Jax.cumsum x;
    let_expr x_id : int [n] := .bind .choice *[x, x_id, 0];
    return x_id

#eval IO.println (indexNonzero (n := 12)).pretty_print
/-
  %0: const int [12] 1 
  %1: const int [12] 0 
  %2: where $0 %0 %1
  %3: cumsum %2
  %4: where %2 %3 %1
  return %4, 
-/

#check Finset.sort

theorem indexNonzero.preserve_zero {n : ℕ} (x : Fin n → ℤ) (i : Fin n) :
    x i = 0 → (indexNonzero.eval Jax.FloatAsReal *[x]).get 0 i = 0 := by
  intro h
  simp [Jax.ExprGroup.eval, indexNonzero, Jax.Expr.eval, Jax.Expr.eval.recursive_eval,
    Jax.TensorImpl.impl, Jax.cumsum, Jax.Tensor.map₃, Jax.Tensor.const, h]

theorem indexNonzero.range {n : ℕ} (x : Fin n → ℤ) :
    let n_nonzero := Nat.card {i | x i ≠ 0};
    let output := (indexNonzero.eval Jax.FloatAsReal *[x]).get 0;
    Set.range output = {i : ℤ | 0 ≤ i ∧ i.natAbs ≤ n_nonzero} := by
  intro m y
  have hy : y = (indexNonzero.eval Jax.FloatAsReal *[x]).get 0 := rfl
  simp [Jax.ExprGroup.eval, indexNonzero, Jax.Expr.eval, Jax.Expr.eval.recursive_eval, Jax.TensorImpl.impl,
    Jax.cumsum, Jax.Tensor.map₃, Jax.Tensor.const] at hy
  rw [hy]
  ext i
  constructor
  · simp
    intro j h
    constructor
    · rw [← h]
      positivity
    · sorry
  · simp
    intro ⟨h₁, h₂⟩
    by_contra!
      

  
