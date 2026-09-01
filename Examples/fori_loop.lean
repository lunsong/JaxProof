import SSA

def power_loop_body {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    return Xla.mul x x

def power_loop {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩,
    m : ⟨.int, []⟩
  begin
    return Xla.fori_loop power_loop_body m x .nil

#eval IO.println (power_loop (n := 10)).code
/-
%0 = repeat; @0, $1, $0
return %0

&0
%0 = mul; $0, $0
return %0
-/

example (m n : ℕ) (x : Fin n → ℝ) :
    (power_loop (n := n)).eval Xla.DirectImpl x (m : ℤ) = Index.single (x ^ (2 ^ m)) := by
  simp only [power_loop, Xla.fori_loop, List.cons_append, List.nil_append, List.length_cons,
    List.length_nil, Nat.reduceAdd, Fin.getElem_fin, power_loop_body, Xla.mul, Xla.bindPrim,
    Fin.zero_eta, Fin.isValue, Fin.mk_one, SSA.Expr.eval, Curry.map, Curry.get, SSA.evalType.bind,
    SSA.Impl.bind, Curry.uncurry, Curry.of, Curry.curry, Curry.transpose, Fin.coe_ofNat_eq_mod,
    Nat.zero_mod, List.getElem_cons_zero, SSA.SimpleImpl.bind, SSA.Tensor.map₂,
    Fin.succ_zero_eq_one, Index.append, Curry.arg, Curry.pure, Curry.map₂, Index.single, id_eq,
    Int.natAbs_natCast]
  ext r
  fin_cases r
  induction m with
  | zero =>
    simp; rfl
  | succ m ih =>
    simp only [List.length_cons, List.length_nil, Nat.reduceAdd, Fin.zero_eta, Fin.isValue,
      Fin.getElem_fin, Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero, Nat.repeat]
    congr
    simp [Index.single] at ih
    simp only [Fin.isValue, ih, Pi.pow_apply]
    ext i
    simp only [Pi.pow_apply]
    rw [pow_succ, pow_mul, pow_two]
