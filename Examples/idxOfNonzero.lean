import Xla

def idxOfNonzero {n : ℕ} :
  Xla.SimpleExpr [⟨.int, [n]⟩] ⟨.int, [n]⟩ :=
  Soir.Expr.ofFn fun x =>
    -- First we find nonzero elements of x
    let x_nonzero := Xla.choice x (Xla.ofNat 1) (Xla.ofNat 0);
    -- Then we give each nonzero element an index by counting the
    -- number of nonzero elements before it
    let x_id := Xla.cumsum x_nonzero;
    Xla.choice x x_id 0

#eval IO.println (idxOfNonzero (n := 12)).code
/-
%0 = const int [12] 1; 
%1 = const int [12] 0; 
%2 = where; $0, %0, %1
%3 = cumsum; %2
%4 = where; $0, %3, %1
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
    idxOfNonzero.eval x = idxOfNonzero_def x := by
  ext i
  simp [idxOfNonzero, reduce_xla, reduce_soir, reduce_tensor, idxOfNonzero_def]
  split_ifs with h
  · rfl
  · rw [Finset.idxOf_sort_of_mem (by simpa)]
    conv_lhs =>
      arg 2; intro j; rw [← ite_not]
    rw [← Finset.sum_filter, Finset.sum_const]
    simp only [Int.nsmul_eq_mul, mul_one]
    conv_lhs =>
      arg 1; arg 1
      equals insert i (Finset.filter (fun j => j < i ∧ x j ≠ 0) Finset.univ) =>
        ext j; grind
    simp; congr 1; ext j; grind
