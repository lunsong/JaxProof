import SSA

def norm_xla {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let x2  := Xla.mul x x;
    let x2_sumed  := Xla.sum 1 x2;
    return Xla.sqrt x2_sumed

def normalize_xla {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let x_norm := norm_xla.apply x;
    let x_norm_broadcasted := Xla.broadcast [⟨n, false⟩] x_norm;
    return Xla.div x x_norm_broadcasted

#eval IO.println (normalize_xla (n := 12)).code
/-
%0 = call; @0, $0
%1 = broadcast [false]; %0
%2 = div; $0, %1
return %2

&0
%0 = mul; $0, $0
%1 = sum 1; %0
%2 = sqrt; %1
return %2
-/

theorem norm_def (n : ℕ) (x : Fin n → ℝ) :
    norm_xla.eval Xla.DirectImpl x = Index.single √(∑ i, (x i)^2) := by
  simp [norm_xla, Xla.sqrt, Xla.bindPrim, SSA.Expr.eval, Curry.map, Curry.get, SSA.evalType.bind,
    SSA.Impl.bind, SSA.SimpleImpl.bind, SSA.Tensor.map]
  congr
  simp [SSA.Expr.eval, Xla.sum, Xla.bindPrim, Curry.map, Curry.get, SSA.Impl.bind,
    SSA.evalType.bind, Index.single, SSA.SimpleImpl.bind, Xla.mul, SSA.Tensor.map₂, Curry.map₂,
    Curry.arg, Curry.pure, Index.append, pow_two]

theorem normalize_def (n : ℕ) (x : Fin n → ℝ) : 
    normalize_xla.eval Xla.DirectImpl x = Index.single (fun i => x i / √(∑ j, (x j)^2)) := by
  simp only [normalize_xla, Xla.div, Xla.bindPrim, List.length_nil, Fin.getElem_fin,
    List.length_cons, Nat.reduceAdd, Fin.zero_eta, Fin.isValue, Xla.broadcast, List.map_cons,
    List.map_nil, List.drop_succ_cons, List.drop_zero, SSA.Expr.eval, Curry.map, Curry.get,
    SSA.evalType.bind, SSA.Impl.bind, SSA.SimpleImpl.bind, SSA.Tensor.map₂, Curry.map₂,
    Index.append, Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero, Curry.arg, Curry.pure,
    Fin.succ_zero_eq_one, SSA.Tensor.broadcast, id_eq, List.nil_append]
  congr
  ext i
  simp [Index.single, norm_def]
