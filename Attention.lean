import JaxProof

open Jax.Api

jax_def (n₁ : ℕ) (n₂ : ℕ) softmax(x):
  y = exp x;
  y' = einsum [n₁, n₂] [[#0, #1]] [#0] [y]; -- sum over the second axis
  y' = rep n₂ y'; -- recover shape
  return y / y'

noncomputable def softmax_def {n₁ n₂ : ℕ} (x : Matrix (Fin n₁) (Fin n₂) ℝ) :
    Matrix (Fin n₁) (Fin n₂) ℝ := 
  Matrix.of fun i j ↦ Real.exp (x i j) / ∑ k, Real.exp (x i k)

#print Finset.Nonempty

#eval IO.println (Jax.trace (softmax 10 10)).code

theorem softmax_eq_def (n₁ n₂ : ℕ) (x : Matrix (Fin n₁) (Fin n₂) ℝ) (hn : n₁ ≠ 0 ∧ n₂ ≠ 0) :
    Jax.native (softmax n₁ n₂) (Jax.Array.ofMatrix x) = Jax.Array.ofMatrix (softmax_def x) := by
  simp [softmax, Jax.Array.ofMatrix, HDiv.hDiv, Jax.Array.div, Jax.Array.rep, Jax.Array.einsum,
    Jax.allFloat, Jax.allFloat.go, Jax.Einsum.sum, Jax.Einsum.prod, Jax.Einsum.prod.go]
  conv_lhs =>
    arg 1
    equals False =>
      simp only [eq_iff_iff, iff_false, not_exists]
      intro idx
      conv =>
        arg 1; lhs; arg 2; intro idx
        conv =>
          arg 2; arg 1; intro i
          equals idx i => 
            congr <;>  fin_cases i <;> simp
        change (List.ofFn (Real.exp ∘ fun i ↦ x i.divNat i.modNat))[idx.flatten.val]
        simp only [List.prod_cons, List.prod_nil, List.getElem_ofFn, Function.comp_apply]
      apply ne_of_gt
      apply Finset.sum_pos
      · intro i h
        exact Real.exp_pos _
      · sorry
  simp only [↓reduceIte, Fin.isValue]
  congr
  ext i
  simp [Div.div, DivInvMonoid.div', softmax_def, div_eq_mul_inv, Jax.ValidIdx.unflatten]
  conv_lhs =>
    arg 1
    equals ({idx' | idx' ⟨0, by simp⟩ = i.divNat} : Finset (Jax.ValidIdx [n₁, n₂])) =>
      congr
      ext idx
      rw [iff_eq_eq]
      congr 1
      rw [← Fin.val_eq_val]
      simp
      rfl
  let f (k : Fin n₂) : Jax.ValidIdx [n₁, n₂] := fun j ↦
    if h : j = ⟨0, by simp⟩ then
      i.divNat.cast <| by
        simp [h]
    else
      k.cast <| by
        have h₁ : j.val < 2 := j.isLt
        have h₂ : j.val ≠ 0 := fun hc ↦ h ((Fin.val_eq_val _ _).mp hc)
        have : j.val = 1 := by omega
        simp[this]
  apply Eq.symm
  apply Finset.sum_of_injOn f
  · intro a ha b hb h
    simp [f] at h
    rw [funext_iff] at h
    specialize h 1
    simpa using h
  · simp [f]
  · intro j hj hj'
    simp [f] at hj' hj
    specialize hj' (j ⟨1, by simp⟩)
    contrapose! hj'
    simp [←hj]
    apply funext
    intro l
    fin_cases l <;> simp
  · simp
    intro j
    conv_rhs =>
      arg 2; arg 1
      equals f j =>
        apply funext
        intro i
        congr <;> fin_cases i <;> simp
    conv_rhs =>
      change (List.ofFn (Real.exp ∘ fun i ↦ x i.divNat i.modNat))[(f j).flatten.val]
      simp only [List.prod_cons, List.prod_nil, List.getElem_ofFn, Function.comp_apply]
    congr 2
    · simp [f, Jax.ValidIdx.flatten]
      rw [← Fin.val_eq_val, Fin.coe_divNat, Fin.coe_divNat, Nat.add_div_of_dvd_right]
      · rw [Nat.mul_div_cancel]
        simp
        omega
      · simp
    · simp [f, Jax.ValidIdx.flatten]
      rw [← Fin.val_eq_val, Fin.coe_modNat]
      simp only [Nat.mul_add_mod_self_right]
      exact (Nat.mod_eq_of_lt j.isLt).symm
    

jax_def (n_head : ℕ) (n_item : ℕ) (dim_qk : ℕ) (dim_v : ℕ) Attention(Q, K, V):
  qk = einsum [n_head, n_item, n_item, dim_qk] [[#1, #0, #3], [#2, #0, #3]] [#0, #1, #2] [Q, K];
  qk = softmax (n_head * n_item) (n_item) qk;
  out = einsum [n_head, n_item, n_item, dim_v] [[#0, #1, #2], [#2, #0, #3]] [#0, #1, #3] [qk, V];
  return out

#eval IO.println (Jax.trace (Attention 10 20 30 30)).code

theorem Attention_equivarient (n_head : ℕ) (n_item : ℕ) (dim_qk : ℕ) (dim_v : ℕ)
  (Q : Matrix (Fin n_item) (Fin (n_head * dim_qk)) ℝ)
  (K : Matrix (Fin n_item) (Fin (n_head * dim_qk)) ℝ)
  (V : Matrix (Fin n_item) (Fin (n_head * dim_v)) ℝ) 
  (σ : Equiv.Perm (Fin n_item)) :
  let f := Jax.native (Attention n_head n_item dim_qk dim_v);
  let q := Jax.Array.ofMatrix Q;
  let k := Jax.Array.ofMatrix K;
  let v := Jax.Array.ofMatrix V;
  let q' := Jax.Array.ofMatrix (Q ∘ σ);
  let k' := Jax.Array.ofMatrix (K ∘ σ);
  let v' := Jax.Array.ofMatrix (V ∘ σ);
  ∃ y : Matrix (Fin n_item) (Fin (n_head * dim_v)) ℝ,
    f q k v = Jax.Array.ofMatrix y ∧ f q' k' v' = Jax.Array.ofMatrix (y ∘ σ) := by
  intro f q k v q' k' v'
  conv =>
    arg 1; intro y; arg 1
    conv =>
      arg 1
      simp [f, Attention, Jax.Array.einsum, Jax.allFloat, Jax.allFloat.go, q, k, Jax.Array.ofMatrix,
        Jax.Einsum.sum, Jax.Einsum.prod, Jax.Einsum.prod.go]
  sorry



