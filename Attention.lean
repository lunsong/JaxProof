import JaxProof

open Jax.Api

jax_def (n₁ : ℕ) (n₂ : ℕ) softmax(x):
  y = exp x;
  y' = einsum [n₁, n₂] [[#0, #1]] 1 [y]; -- sum over the second axis
  y' = rep n₁ y'; -- recover shape
  return y / y'

noncomputable def softmax_def {n₁ n₂ : ℕ} (x : Matrix (Fin n₁) (Fin n₂) ℝ) :
    Matrix (Fin n₁) (Fin n₂) ℝ := 
  Matrix.of fun i j ↦ Real.exp (x i j) / ∑ k, Real.exp (x i k)

#print Finset.Nonempty

#eval IO.println (Jax.trace (softmax 11 10)).code

theorem softmax_eq_def (n₁ n₂ : ℕ) (x : Matrix (Fin n₁) (Fin n₂) ℝ) (hn : n₁ ≠ 0 ∧ n₂ ≠ 0) :
    Jax.native (softmax n₁ n₂) (Jax.Array.ofMatrix x) = Jax.Array.ofMatrix (softmax_def x) := by
  simp? [softmax, Jax.Array.ofMatrix, Jax.Array.ofTensor]
  conv_lhs =>
    fun
    simp [HDiv.hDiv]
  simp [Jax.Array.div, Jax.Array.rep, Jax.Array.einsum, Jax.List.map?.go, Jax.Array.toTensor,
    Jax.Array.ofTensor, mul_comm n₁ n₂]
  split_ifs with h
  · contrapose! h
    intro i
    simp [Jax.Tensor.einsum, Jax.Tensor.einprod, Jax.Tensor.unflatten, Jax.Tensor.flatten]
    intro hc
    simp [Jax.Tensor] at hc
    sorry
  apply congrArg
  apply congrArg
  ext i
  --simp [softmax_def, Jax.Tensor.einsum, Jax.Tensor.einprod, Jax.Tensor.unflatten, Jax.Tensor.flatten]
  conv_lhs =>
    arg 2
    simp [Jax.Tensor.einsum, Jax.Tensor.einprod]
    arg 1
    arg 2
    intro i i'
    simp [Jax.Tensor.flatten, Jax.Tensor.unflatten]
  conv_rhs =>
    simp [softmax_def, Jax.Tensor.flatten]
  apply congrArg
  conv_lhs =>
    change (∑ i, fun i' ↦ Real.exp (x i i')) i.divNat









  

