import SSA

def matmul {n m l : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n, m]⟩,
    y : ⟨.float, [m, l]⟩
  begin
    let_expr x : [⟨.float, [m, n]⟩] := Xla.transpose [0,1].formPerm x;
    let_expr z : [⟨.float, [n, l]⟩] := Xla.dot_general [] [m] [n] [l] x y;
    return z

#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code
/-
%0 = transpose [1, 0]; $0
%1 = dot_general 1 0; %0, $1
return %1
-/

example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Xla.DirectImpl x y = Index.single (x * y) := by
  simp only [matmul, Xla.dot_general, Xla.bindPrim, List.cons_append, List.nil_append,
    List.length_nil, Fin.getElem_fin, List.length_cons, Nat.reduceAdd, List.formPerm_cons_cons,
    List.formPerm_singleton, Fin.zero_eta, Fin.isValue, Fin.mk_one, SSA.Expr.eval, Curry.map,
    Curry.get, SSA.evalType.bind, SSA.Impl.bind, SSA.SimpleImpl.bind, List.drop_succ_cons,
    List.drop_zero, SSA.Tensor.map, Curry.pure, Curry.map₂, Index.append, SSA.Tensor.uncurry',
    id_eq, Fin.succ_zero_eq_one, Fin.coe_ofNat_eq_mod, Nat.reduceMod, List.getElem_cons_succ,
    List.getElem_cons_zero, SSA.Tensor.cast_rfl, SSA.Tensor.map₂, SSA.Tensor.sumN,
    SSA.Tensor.sumFirst, List.tail_cons]
  congr
  ext i j
  simp [Matrix.mul_apply]
  congr
