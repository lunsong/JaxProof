import SSA

def A : Fin 3 → Fin 3 := ![2,0,1]

def Coulomb {n_atom : ℕ} :=
  ssa Xla.XlaOp with
    x  : ⟨.float, [n_atom, 3]⟩
  begin
    let sender   := Xla.broadcast [⟨n_atom,  true⟩, ⟨n_atom, false⟩, ⟨3, true⟩] x;
    let receiver := Xla.broadcast [⟨n_atom, false⟩, ⟨n_atom,  true⟩, ⟨3, true⟩] x;
    let r := Xla.sub sender receiver;
    let r2 := Xla.mul r r;
    let σ : Equiv.Perm (Fin 3) := {
      toFun := ![2, 0, 1]
      invFun := ![1, 2, 0]
      right_inv := by decide
      left_inv := by decide
    };
    let r2 := Xla.transpose (s := [n_atom, n_atom, 3]) σ r2;
    let d := Xla.div 1 (Xla.sqrt (Xla.sum 1 r2));
    return Xla.sum 2 d

#eval IO.println (Coulomb (n_atom := 12)).code

open SSA in
example {n_atom : ℕ} (x : Fin n_atom → Fin 3 → ℝ) :
    Coulomb.eval Xla.DirectImpl x = Index.single (∑ i, ∑ j, 1 / √(∑ k, (x i k - x j k) ^ 2)) := by
  ext r
  fin_cases r
  simp only [List.length_cons, List.length_nil, Nat.reduceAdd, Fin.isValue, Equiv.coe_fn_mk,
    Fin.getElem_fin, Fin.zero_eta, Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero,
    Coulomb, Xla.sum, Xla.bindPrim, Xla.div, Xla.sqrt, Xla.transpose, Xla.mul, List.map_cons,
    List.map_nil, Xla.sub, Xla.broadcast, Expr.eval, Curry.map, Curry.get, evalType.bind, Impl.bind,
    Index.single, SimpleImpl.bind, Curry.map₂, Index.append, Fin.succ_zero_eq_one, Tensor.transpose,
    List.get_eq_getElem, Tensor.map₂, Tensor.broadcast, Curry.arg, Curry.pure, id_eq,
    List.nil_append, Equiv.coe_fn_symm_mk, Matrix.cons_val_zero, Fin.cast_eq_self,
    Matrix.cons_val_one, Nat.reduceMod, List.getElem_cons_succ, Fin.succ_one_eq_two,
    Matrix.cons_val, Tensor.sumN, Tensor.sumFirst, Fin.mk_one, Fin.reduceFinMk, List.tail_cons,
    one_div]
  conv_lhs =>
    arg 2; intro i j
    conv =>
      arg 1
      conv =>
        arg 2
        simp [OfNat.ofNat]
      simp [Expr.eval, Xla.bindPrim, Curry.map, Curry.get, evalType.bind, Impl.bind, Index.single,
        SimpleImpl.bind, Curry.pure]
    arg 2; arg 2; arg 2; intro r; arg 1; intro idx 
    rw [← pow_two]
    change (x (idx (1 : Fin 3)) (idx (0 : Fin 3)) - x (idx (2 : Fin 3)) (idx (0 : Fin 3)))^2
  have {n m : ℕ} (x : Tensor ℝ [n, m]) : x.sumN 2 = ∑ j, ∑ i, x i j := by simp [Tensor.sumN]
  simp only [Fin.isValue, Curry.of, Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero,
    Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero, Fin.reduceFinMk, Matrix.cons_val,
    Nat.reduceMod, List.getElem_cons_succ, one_div, this]
  congr
  ext i
  congr
  ext j
  congr
  simp only [Tensor.map, Fin.isValue, List.ofFn, Fin.foldr, Fin.foldr.loop, Fin.reduceFinMk,
    Matrix.cons_val, Fin.coe_ofNat_eq_mod, Nat.reduceMod, List.getElem_cons_succ,
    List.getElem_cons_zero, Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero, Nat.zero_mod,
    Fin.zero_eta, List.drop_succ_cons, List.drop_zero, Curry.map, Finset.sum_apply]
  congr
  ext k
  ring
