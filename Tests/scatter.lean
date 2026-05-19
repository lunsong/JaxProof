import SSA

def permInv {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.int, [n]⟩
  begin
    return Xla.scatter (Xla.const_int [n] 0) (Xla.iota n) x

#eval IO.println (permInv (n := 10)).code
/-
%0 = const int [10] 0; 
%1 = iota 10; 
%2 = scatter; %0, %1, $0
return %2
-/



example (n : ℕ) (hn : n ≠ 0) (σ : Equiv.Perm (Fin n)) :
    permInv.eval Xla.DirectImpl (fun i => σ i) = Index.single (fun i => (σ.symm i : ℤ)) := by
  simp only [permInv, Xla.scatter, List.length_cons, List.length_nil, Nat.reduceAdd,
    List.replicate_one, Xla.bindPrim, Fin.getElem_fin, List.cons_append, List.nil_append, Xla.iota,
    Curry.get, Fin.zero_eta, Fin.isValue, Curry.of, SSA.Expr.eval, Curry.map, SSA.evalType.bind,
    SSA.Impl.bind, SSA.SimpleImpl.bind, Curry.map₂, Index.append, Fin.succ_zero_eq_one,
    Fin.succ_one_eq_two]
  conv_lhs =>
    arg 1; arg 1
    change (0 : Fin n → ℤ)
  congr
  simp only [Xla.DirectImpl.scatter, List.length_cons, List.length_nil, Nat.reduceAdd,
    Fin.getElem_fin, Fin.val_eq_zero, List.getElem_cons_zero, ne_eq, hn, not_false_eq_true,
    implies_true, ↓reduceDIte, List.replicate_one, Curry.of, Index.replicate, Fin.zero_eta,
    Fin.isValue, Index.single, Curry.get, Pi.zero_apply, SSA.Expr.join, SSA.Expr.eval, Curry.map₂,
    Index.append, Curry.map, Curry.arg, Curry.pure]
  have (α : Fin 1 → Type) (f g : (i : Fin 1) → α i) : f = g ↔ f 0 = g 0 := by
    constructor
    · intro h; rw [h]
    · intro h; ext i; fin_cases i; exact h
  conv_lhs =>
    intro i; arg 2; arg 1; intro j; arg 1
    rw [this]
    simp
  ext i
  have : NeZero n := ⟨hn⟩
  have : Fin.find? (fun j => Fin.intCast (σ j : ℤ) = i) = some (σ.symm i) := by
    simp only [Fin.intCast, Nat.cast_nonneg, ↓reduceIte, Int.natAbs_natCast, Fin.ofNat_eq_cast,
      Fin.cast_val_eq_self, Fin.findSome?_eq_some_iff, Option.guard_eq_some_iff, decide_eq_true_eq,
      Option.guard_eq_none_iff, decide_eq_false_iff_not, ↓existsAndEq, Equiv.apply_symm_apply,
      and_self, true_and]
    intro j hj hc
    apply hj.ne
    rwa [Equiv.eq_symm_apply]
  -- `simp [this]` would raise: Don't know how to synthesize place holder
  rw [this]

