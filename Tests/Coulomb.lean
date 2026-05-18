import SSA

def A : Fin 3 → Fin 3 := ![2,0,1]

def diag_mask {n : ℕ} :=
  ssa Xla.XlaOp with
  begin
    let_expr idx : [⟨.int, [n]⟩] := Xla.iota;
    let i := Xla.broadcast [⟨n, true⟩, ⟨n, false⟩] idx;
    let j := Xla.broadcast [⟨n, false⟩, ⟨n, true⟩] idx;
    return Xla.eq i j

theorem diag_mask_def {n : ℕ} (i j : Fin n) :
    Xla.simpleEval diag_mask i j = if i = j then (1 : ℤ) else 0 := by
  simp only [List.map_nil, Xla.simpleEval, Curry.map, List.map_cons, diag_mask, Xla.eq,
    Xla.bindPrim, List.length_nil, Fin.getElem_fin, Xla.broadcast, Xla.iota, Fin.isValue,
    SSA.Expr.eval, Curry.get, SSA.evalType.bind, SSA.Impl.bind, Index.single, List.length_cons,
    Nat.reduceAdd, SSA.SimpleImpl.bind, SSA.Tensor.map₂, Curry.map₂, Index.append,
    SSA.Tensor.broadcast, Fin.zero_eta, id_eq, Fin.succ_zero_eq_one, List.nil_append, Nat.cast_inj]
  conv_lhs =>
    arg 3; intro h i j; arg 1
    equals i = j =>
      simp [Fin.val_eq_val]
  rfl

example : Function.Injective (![2, 0, 1] : Fin 3 → Fin 3) := by decide

def Coulomb {n_atom : ℕ} :=
  ssa Xla.XlaOp with
    x  : ⟨.float, [n_atom, 3]⟩
  begin
    let sender   := Xla.broadcast [⟨n_atom,  true⟩, ⟨n_atom, false⟩, ⟨3, true⟩] x;
    let receiver := Xla.broadcast [⟨n_atom, false⟩, ⟨n_atom,  true⟩, ⟨3, true⟩] x;
    let r := Xla.sub sender receiver;
    let r2 := Xla.mul r r;
    let r2 := Xla.transpose' r2 (![2, 0, 1] : Fin 3 → Fin 3);
    let d := Xla.div 1 (Xla.sqrt (Xla.sum 1 r2));
    let_expr mask : [⟨.int, [n_atom, n_atom]⟩] := diag_mask.apply .nil;
    let d := Xla.choice mask 0 d;
    return Xla.sum 2 d

#eval IO.println (Coulomb (n_atom := 12)).code

open SSA in
example {n_atom : ℕ} (x : Fin n_atom → Fin 3 → ℝ) :
    Xla.simpleEval Coulomb x = ∑ i, ∑ j with i ≠ j, 1 / √(∑ k, (x i k - x j k) ^ 2) := by
  have := diag_mask_def (n := n_atom)
  simp only [List.map_nil, Xla.simpleEval, Curry.map, List.map_cons, Fin.isValue] at this
  simp only [List.drop_succ_cons, List.drop_zero, Xla.simpleEval, Curry.map, Coulomb, Xla.sum,
    Xla.bindPrim, List.length_nil, Fin.getElem_fin, Xla.choice, List.cons_append, List.nil_append,
    List.map_cons, List.map_nil, Xla.div, List.length_cons, Nat.reduceAdd, Fin.isValue, Xla.sqrt,
    Xla.transpose', Xla.mul, Xla.sub, Xla.broadcast, Fin.zero_eta, Expr.eval, Curry.get,
    evalType.bind, Impl.bind, Index.single, SimpleImpl.bind, Tensor.sumN, Tensor.sumFirst,
    Tensor.map₃, Curry.map₂, Index.append, this, bne_iff_ne, ne_eq, ite_eq_right_iff, one_ne_zero,
    imp_false, Decidable.not_not, Fin.succ_zero_eq_one, Fin.succ_one_eq_two, Tensor.map,
    Tensor.transpose, List.get_eq_getElem, Tensor.map₂, Tensor.broadcast, Curry.arg, Curry.pure,
    id_eq, Fin.cast_eq_self, Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero,
    Nat.reduceMod, List.getElem_cons_succ, Curry.of, Fin.mk_one, Fin.reduceFinMk,
    Matrix.cons_val_one, Matrix.cons_val_zero, Matrix.cons_val, List.tail_cons, Finset.sum_apply,
    one_div]
  conv_lhs =>
    arg 3; intro h; arg 2; intro i
    conv =>
      arg 2; intro j
      conv =>
        arg 2
        simp [OfNat.ofNat, Expr.eval, Xla.bindPrim, Curry.map, Curry.get, evalType.bind,
          Impl.bind, SimpleImpl.bind]
        change 0
      conv =>
        arg 3; arg 1
        simp [OfNat.ofNat, Expr.eval, Xla.bindPrim, Curry.map, Curry.get, evalType.bind,
          Impl.bind, SimpleImpl.bind]
        change 1
      conv =>
        arg 3; arg 2; arg 5
        conv =>
          arg 2; intro a b c
          rw [← pow_two]
      conv =>
        arg 3; arg 2
        change √(∑ a, (x j a - x i a) ^ 2)
        equals √(∑ a, (x i a - x j a) ^ 2) =>
          congr
          ext a
          exact sub_sq_comm (x j a) (x i a)
    simp [Finset.sum_ite]
    conv =>
      arg 1
      equals {x | ¬ i = x} =>
        ext a
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact Iff.intro Ne.symm Ne.symm
  rfl
    

