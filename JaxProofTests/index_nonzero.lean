import JaxProof

def idxOfNonzero {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    -- First we find nonzero elements of x
    let x_nonzero := Jax.choice x 1 0;
    -- Then we give each nonzero element an index by counting the
    -- number of nonzero elements before it
    let x_id := Jax.cumsum x_nonzero;
    return Jax.choice x x_id 0

#eval IO.println (idxOfNonzero (n := 12)).pretty_print
/-
  %0: const int [12] 1 
  %1: const int [12] 0 
  %2: where $0 %0 %1
  %3: cumsum %2
  %4: where %2 %3 %1
  return %4, 
-/

def idxOfNonzero_def {n : ℕ} (x : Fin n → ℤ) (i : Fin n) : ℤ :=
  if h : x i = 0 then 0 else List.idxOf i (Finset.sort {i | x i ≠ 0}) + 1

theorem Finset.idxOf_sort_of_mem {m : ℕ} {s : Finset (Fin m)} {x : (Fin m)} :
    x ∈ s → s.sort.idxOf x = Finset.card {y ∈ s| y < x} := by
  intro h
  let i := s.sort.idxOf x
  have h_mono : s.sort.SortedLT := Finset.sortedLT_sort _
  rw [List.sortedLT_iff_strictMono_get] at h_mono
  have h1 : ∀ y, y ∈ s.sort.take i → y ∈ s ∧ y < x := by
    intro y hy'
    have hy : y ∈ s.sort := List.mem_of_mem_take hy'
    constructor
    · simpa using hy
    simp only [List.mem_take_iff_idxOf_lt hy] at hy'
    let j := s.sort.idxOf y
    change j < i at hy'
    let i' : Fin s.sort.length := .mk i <| by
      simp only [i, List.idxOf_lt_length_iff]
      simpa
    let j' : Fin s.sort.length := .mk j <| by
      simp only [j, List.idxOf_lt_length_iff]
      exact hy
    have : j' < i' := by simp [i', j', hy']
    specialize h_mono this
    simpa [i', j', i, j] using h_mono
  have h2 : ∀ y ∈ s, y < x → y ∈ s.sort.take i := by
    intro y h₁ h₂
    rw [List.mem_take_iff_idxOf_lt (by simpa)]
    unfold i
    by_contra!
    contrapose! h₂
    have hx : s.sort.idxOf x < s.sort.length := by
      rw [List.idxOf_lt_length_iff]
      simpa
    have hy : s.sort.idxOf y < s.sort.length := by
      rw [List.idxOf_lt_length_iff]
      simpa
    rw [← List.idxOf_get hx, ← List.idxOf_get hy]
    apply h_mono.monotone
    exact this
  have : (s.sort.take i).toFinset = {y ∈ s | y < x} := by
    ext y
    constructor
    · simpa using h1 y
    · simpa using h2 y
  rw [← this]
  simp only [List.card_toFinset]
  rw [List.dedup_eq_self.mpr]
  · simp only [List.length_take, left_eq_inf, ge_iff_le, i]
    exact List.idxOf_le_length
  apply List.Nodup.sublist (List.take_sublist _ _)
  simp


theorem idxOfNonzero_eq_def {n : ℕ} {x : Fin n → ℤ} :
    idxOfNonzero.eval Jax.FloatAsReal *[x] = *[idxOfNonzero_def x] := by
  simp only [idxOfNonzero, Fin.isValue, Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl,
    Jax.Expr.eval.recursive_eval, Jax.DList.get_zero_cons, CharP.cast_eq_zero,
    Jax.Tensor.map₃, bne_iff_ne, ne_eq, ite_not, Jax.DList.cons.injEq, and_true, Jax.choice]
  apply funext
  intro i
  simp only [Jax.Tensor.const, Fin.isValue, idxOfNonzero_def, ne_eq, dite_eq_ite]
  by_cases h : x i = 0
  · simp [h]
  · simp only [h, ↓reduceIte, Jax.cumsum, Fin.isValue, Jax.Expr.eval, Jax.TensorImpl.impl,
    Jax.Expr.eval.recursive_eval, Jax.DList.get_zero_cons, Jax.Tensor.const, Nat.cast_one,
    CharP.cast_eq_zero, Jax.Tensor.map₃, bne_iff_ne, ne_eq, ite_not, Jax.Tensor.cumsum]
    rw [Finset.idxOf_sort_of_mem (by simpa), Finset.sum_filter]
    conv_lhs =>
      arg 2; intro j
      equals if j ≤ i ∧ x j ≠ 0 then (Jax.Tensor.const 1) else 0 =>
        rw [ite_and, ite_not]
        rfl
    rw [← Finset.sum_filter]
    simp only [ne_eq, Jax.Tensor.const, Finset.sum_const]
    conv_lhs =>
      arg 1; arg 1
      equals Finset.filter (· < i) {j | x j ≠ 0} ∪ {i} =>
        ext j
        constructor
        · simp only [Finset.mem_filter, Finset.mem_univ, true_and, ne_eq, Finset.union_singleton,
          Finset.mem_insert, and_imp]
          intro h1 h2
          rcases lt_or_eq_of_le h1 with h1 | h1
          · exact .inr ⟨h2, h1⟩
          · exact .inl h1
        · simp only [ne_eq, Finset.union_singleton, Finset.mem_insert, Finset.mem_filter,
          Finset.mem_univ, true_and]
          intro h1
          rcases h1 with h1 | h1
          · exact ⟨h1.le, h1 ▸ h⟩
          · exact ⟨h1.2.le, h1.1⟩
    simp only [ne_eq, Finset.union_singleton, Finset.mem_filter, Finset.mem_univ, true_and,
      lt_self_iff_false, and_false, not_false_eq_true, Finset.card_insert_of_notMem]
    simp only [HSMul.hSMul, SMul.smul, AddMonoid.nsmul, Nat.repeat]
    congr
    set m := Finset.card {x ∈ {j | ¬x j = 0} | x < i}
    induction m with
    | zero => rfl
    | succ m ih =>
      simp [ih, Nat.repeat]
