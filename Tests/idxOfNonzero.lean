import SSA

def idxOfNonzero {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.int, [n]⟩
  begin
    -- First we find nonzero elements of x
    let_expr x_nonzero : [⟨.int, [n]⟩] := Xla.choice x 1 0;
    -- Then we give each nonzero element an index by counting the
    -- number of nonzero elements before it
    let x_id := Xla.cumsum x_nonzero;
    return Xla.choice x x_id 0

#eval IO.println (idxOfNonzero (n := 12)).code
/-
%0 = const int [12] 1;;
%1 = const int [12] 0;;
%2 = where;;$0;%0;%1
%3 = cumsum;;%2
%4 = where;;$0;%3;%1
return %4
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
    idxOfNonzero.eval Xla.DirectImpl x = Index.single (idxOfNonzero_def x) := by
  simp only [idxOfNonzero, Xla.choice, Xla.bindPrim, List.length_nil, Fin.getElem_fin,
    List.cons_append, List.nil_append, List.length_cons, Nat.reduceAdd, Fin.zero_eta, Fin.isValue,
    Xla.cumsum, SSA.Expr.eval, Curry.map, Curry.get, SSA.evalType.bind, SSA.Impl.bind,
    SSA.SimpleImpl.bind, SSA.Tensor.map₃, Curry.map₂, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
    List.getElem_cons_zero, SSA.Tensor.cumsum, bne_iff_ne, ne_eq, Fin.succ_zero_eq_one,
    Fin.succ_one_eq_two, ite_not]
  congr
  ext i
  simp only [Index.append, Index.single, Curry.arg, Curry.pure, List.nil_append, List.length_cons,
    List.length_nil, Nat.reduceAdd, Fin.zero_eta, Fin.isValue, Fin.succ_zero_eq_one,
    idxOfNonzero_def, ne_eq, dite_eq_ite]
  split_ifs with h
  · rfl
  rw [Finset.idxOf_sort_of_mem (by simpa)]
  conv_lhs =>
    arg 2; intro j
    conv =>
      arg 2
      change 0
    conv =>
      arg 3
      change 1
    equals if x j ≠ 0 then 1 else 0 =>
      simp
  rw [Finset.sum_ite]
  simp only [ne_eq, Finset.sum_const, Int.nsmul_eq_mul, mul_one,
    Decidable.not_not, add_zero, mul_zero]
  conv_lhs =>
    arg 1; arg 1
    equals ({y ∈ {i | ¬x i = 0} | y < i} : Finset (Fin n)) ∪ {i} =>
      ext j
      constructor
      · simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.union_singleton,
        Finset.mem_insert, and_imp]
        intro h1 h2
        rcases eq_or_lt_of_le h1 with h1 | h1
        · left; exact h1
        · right; exact ⟨h2, h1⟩
      · simp only [Finset.union_singleton, Finset.mem_insert, Finset.mem_filter, Finset.mem_univ,
        true_and]
        intro h1
        rcases h1 with h1 | h1
        · simp [h1, h]
        · exact ⟨h1.2.le, h1.1⟩
  simp
