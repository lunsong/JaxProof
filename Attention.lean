import JaxProof

open Jax.Api

jax_def (n₁ : ℕ) (n₂ : ℕ) softmax(x):
  y = exp x;
  y' = einsum [n₁, n₂] [[#0, #1]] 1 [y]; -- sum over the second axis
  y' = rep n₁ y'; -- recover shape
  return y / y'

noncomputable def softmax_def {n₁ n₂ : ℕ} (x : Matrix (Fin n₁) (Fin n₂) ℝ) :
    Matrix (Fin n₁) (Fin n₂) ℝ := 
  Matrix.of fun i j ↦ Real.exp (x i j) / ∑ k, Real.exp (x k j)

#print Finset.Nonempty

#eval IO.println (Jax.trace (softmax 11 10)).code

theorem softmax_eq_def (n₁ n₂ : ℕ) (x : Matrix (Fin n₁) (Fin n₂) ℝ) (hn : n₁ ≠ 0 ∧ n₂ ≠ 0) :
    Jax.native (softmax n₁ n₂) (Jax.Array.ofMatrix x) = Jax.Array.ofMatrix (softmax_def x) := by
  simp [softmax, Jax.Array.ofMatrix, Jax.Array.ofTensor]
  conv_lhs =>
    fun
    simp only [HDiv.hDiv]
  simp [Jax.Array.div, Jax.Array.rep, Jax.Array.einsum, Jax.List.map?.go, Jax.Array.toTensor,
    Jax.Array.ofTensor, mul_comm n₁ n₂]
  apply congrArg
  apply congrArg
  ext i
  conv_rhs =>
    simp [Jax.Tensor.flatten, softmax_def]
  conv_lhs =>
    arg 1; simp [Jax.Tensor.flatten]
  apply congrArg
  conv_lhs =>
    arg 1;
    simp [Jax.Tensor.einsum, Jax.Tensor.einprod, Jax.Tensor.unflatten, Jax.Tensor.flatten]
    change (∑ i, fun j ↦ Real.exp (x i j) : Fin n₂ → ℝ)
    rw [Finset.sum_fn (s := Finset.univ) (g := fun i j ↦ Real.exp (x i j))]
  conv_lhs =>
    unfold Jax.Tensor.flatten
    unfold Jax.Tensor.flatten
  simp [Fin.modNat, Fin.divNat]















  

