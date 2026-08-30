import SSA

def matmul {n m l : ℕ}
  : SSA.Expr Xla.XlaOp
    [⟨.float, [n, m]⟩, ⟨.float, [m, l]⟩]
    [⟨.float, [n, l]⟩] :=
  SSA.Expr.ofFn fun x y =>
    let x := Xla.transpose [0, 1].formPerm x;
    Xla.dot_general [] [m] [n] [l] x y

#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code
/-
%0 = transpose [1, 0]; $0
%1 = dot_general 1 0; %0, $1
return %1
-/

example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Xla.DirectImpl x y = Index.single (x * y) := by
  simp only [matmul, SSA.Expr.ofFn, Curry.get, Xla.dot_general, Xla.bindPrim, List.cons_append,
    List.nil_append, List.length_nil, Fin.getElem_fin, List.length_cons, Nat.reduceAdd,
    List.formPerm_cons_cons, List.formPerm_singleton, Fin.zero_eta, Fin.isValue,
    Fin.succ_zero_eq_one, SSA.Expr.eval, Curry.map, SSA.evalType.bind, SSA.Impl.bind,
    SSA.SimpleImpl.bind, List.drop_succ_cons, List.drop_zero, SSA.Tensor.map, Curry.pure,
    Curry.map₂, Index.append, SSA.Tensor.uncurry', id_eq, Fin.coe_ofNat_eq_mod, Nat.reduceMod,
    List.getElem_cons_succ, List.getElem_cons_zero, SSA.Tensor.cast_rfl, SSA.Tensor.map₂,
    SSA.Tensor.sumN, SSA.Tensor.sumFirst, List.tail_cons]
  congr
  refine SSA.Tensor.ext fun i => SSA.Tensor.ext fun j => ?_
  simp only [reduce_xla, SSA.Expr.eval, Curry.map, Curry.get, Curry.arg, Index.single,
    SSA.evalType.bind, SSA.Impl.bind, SSA.SimpleImpl.bind, Curry.pure, SSA.Tensor.curry']
  conv_lhs =>
    change (∑ k, fun i j ↦ x i k * y k j) i j
  simp [Finset.sum_apply, Matrix.mul_apply]
