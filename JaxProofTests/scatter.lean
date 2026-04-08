import JaxProof

def inv_permutation {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    return .bind .scatter *[0, Jax.iota, x]

#eval IO.println (inv_permutation (n := 10)).pretty_print
/-
%0: const int [10] 0 
%1: iota 10 
%2: scatter %0 %1 $0
return %2, 
-/


example (n : ℕ) (σ : Equiv.Perm (Fin n)) :
    let x : Jax.FloatAsReal ⟨.int, [n]⟩ := (Int.ofNat ∘ Fin.val ∘ σ);
    let y : Jax.FloatAsReal ⟨.int, [n]⟩ := (Int.ofNat ∘ Fin.val ∘ σ.symm);
    inv_permutation.eval _ *[x] = *[y] :=
  match n with
  | 0 => by
    intro x y
    simp only [inv_permutation, List.length_cons, List.length_nil, Nat.reduceAdd,
      List.replicate_one, List.replicate_zero, Fin.isValue, Jax.ExprGroup.eval,
      Jax.DList.cons.injEq, and_true]
    ext i
    nomatch i
  | n + 1 => by
    intro x y
    simp only [inv_permutation, List.length_cons, List.length_nil, Nat.reduceAdd,
      List.replicate_one, List.replicate_zero, Fin.isValue, Jax.ExprGroup.eval,
      Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval,
      Jax.FloatAsReal.scatter, Fin.getElem_fin, Fin.val_eq_zero, List.getElem_cons_zero,
      ne_eq, Nat.add_eq_zero_iff, one_ne_zero, and_false, not_false_eq_true, implies_true,
      ↓reduceDIte, Function.comp_apply, Jax.DList.cons.injEq, and_true]
    rw [← Jax.Tensor.of_get (x := y)]
    congr
    ext idx
    let indices :=
      (Jax.ValidIdx.intCast (s := [n + 1]) (by simp)) ∘
        Function.swap (Jax.DList.unfold_replicate (n := 1) *[x])
    let j := Fin.find? fun i => idx = indices i
    conv_lhs =>
      arg 2
      change j
    have : j = some (σ.symm (idx 0)) := by
      simp only [Fin.isValue, Fin.findSome?_eq_some_iff, Option.guard_eq_some_iff,
        decide_eq_true_eq, Option.guard_eq_none_iff, decide_eq_false_iff_not, ↓existsAndEq,
        true_and, j]
      have : NeZero ([n + 1].get 0) := .mk <| by simp
      constructor
      · apply funext
        intro r
        rw [show r = 0 by simp; omega]
        change idx 0 = Fin.intCast (σ (σ.symm (idx 0)))
        simp [Fin.intCast]
      · intro k hk hc
        rw [funext_iff] at hc
        specialize hc 0
        change idx 0 = Fin.intCast (σ k) at hc
        simp [Fin.intCast] at hc
        apply hk.ne
        rw [Equiv.eq_symm_apply]
        exact hc.symm
    rw [this]
    rfl
