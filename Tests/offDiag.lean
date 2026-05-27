import SSA

lemma sq_sub_one (n : ℕ) : n * n - 1 = (n - 1) * (n + 1) := by
  by_cases h : n = 0
  · rw [h]
  rw [Nat.sub_mul, mul_add, mul_one, one_mul, ← Nat.sub_sub, Nat.add_sub_cancel]

def offDiag {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n, n]⟩
  begin
    let x := Xla.flatten x;
    let x := Xla.gather x (Xla.iota (n * n - 1));
    let x := Xla.cast (show [n * n - 1] = [(n - 1) * ((n + 1) * 1)] by congr; simp[sq_sub_one]) x;
    let x := Xla.unflatten [n - 1, n + 1] x;
    let_expr i : [⟨.int, [n - 1, n]⟩] :=
      Xla.broadcast [⟨n - 1, true⟩, ⟨n, false⟩] (Xla.iota (n - 1));
    let_expr j : [⟨.int, [n - 1, n]⟩] :=
      Xla.add (Xla.broadcast [⟨n - 1, false⟩, ⟨n, true⟩] (Xla.iota n)) 1;
    let x := Xla.gather x (i.append j);
    return x

#eval IO.println (offDiag (n := 8)).code

def offDiag_def {n : ℕ} (x : SSA.Tensor ℝ [n, n]) : SSA.Tensor ℝ [n - 1, n] :=
  fun i j =>
    let k : ℕ := i.val * (n + 1) + j + 1
    let i' : Fin n := .mk (k / n) <| by
      rw [Nat.div_lt_iff_lt_mul (Nat.zero_lt_of_lt j.isLt), ← Nat.succ_le_iff]
      have hi := Nat.succ_le_iff.mpr i.isLt
      have hj := Nat.succ_le_iff.mpr j.isLt
      unfold k
      calc
        (i.val * (n + 1) + j.val + 1).succ = i.val.succ * (n + 1) + j.val.succ - n := by grind
        _ ≤ (n - 1) * (n + 1) + j.val.succ - n := by gcongr
        _ ≤ (n - 1) * (n + 1) + n - n := by gcongr
        _ ≤ n * n := by rw [← sq_sub_one, Nat.add_sub_cancel]; grind
    let j' : Fin n := .mk (k % n) (Nat.mod_lt _ (Nat.zero_lt_of_lt j.isLt))
    x i' j'

theorem Index.single_zero {ι : Type} {m : ι → Type} {i : ι} {x : m i} : Index.single x 0 = x := rfl

theorem offDiag_eq_def {n : ℕ} : Xla.simpleEval (offDiag (n := n)) = offDiag_def := by
  ext x i j
  have h1 : n - 1 ≠ 0 := (Nat.zero_lt_of_lt i.isLt).ne.symm
  have h2 : n ≠ 0 := (Nat.zero_lt_of_lt j.isLt).ne.symm
  simp only [Xla.simpleEval, offDiag, Xla.gather, Xla.bindPrim, List.length_cons,
    List.length_nil, Nat.reduceAdd, List.reduceReplicate, Fin.getElem_fin, Xla.unflatten,
    List.prod_cons, List.prod_nil, Xla.cast, List.replicate_one, Xla.flatten, Fin.zero_eta,
    Fin.isValue, Xla.iota, List.replicate_zero, Xla.broadcast, List.map_cons, List.map_nil, Xla.add,
    SSA.Impl.bind, SSA.SimpleImpl.bind,
    Xla.DirectImpl.gather, Curry.of, h1, ↓reduceDIte, Nat.add_eq_zero_iff, h2, one_ne_zero,
    and_self, Index.append, SSA.Tensor.unflatten, mul_one, mul_eq_zero,
    or_self, SSA.Tensor.flatten, Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero,
    Curry.arg, Fin.divNat, List.foldr_cons, List.foldr_nil, Fin.modNat, Nat.div_one,
    Index.single_zero, Fin.intCast, Fin.succ_zero_eq_one, List.nil_append, Nat.cast_nonneg,
    ↓reduceIte, Int.natAbs_natCast, Fin.ofNat_eq_cast, Fin.val_natCast, dvd_mul_right,
    Nat.mod_mod_of_dvd, Fin.mulAdd, add_zero, SSA.Tensor.broadcast, id_eq, Fin.cast_val_eq_self,
    Fin.succ_one_eq_two, SSA.Tensor.map₂, reduce_ssa]
  congr
  simp only [Fin.cast, Fin.isValue]
  congr 1
  · simp only [Fin.isValue, Fin.mk.injEq]
    congr
    conv_lhs =>
      arg 1; arg 2; arg 1
      simp [OfNat.ofNat, Xla.bindPrim, SSA.Expr.eval, Curry.map, Curry.get, SSA.evalType.bind,
        SSA.Impl.bind, SSA.SimpleImpl.bind, Curry.pure, Index.single_zero]
      change
        if 0 ≤ (j.val : ℤ) + 1 then
          Fin.ofNat (n + 1) (Int.natAbs (j.val + 1))
        else
          -(Fin.ofNat (n + 1) (Int.natAbs (j.val + 1)))
      simp [show 0 ≤ (j.val : ℤ) + 1 by omega]
    norm_cast
    rw [Fin.val_natCast, Nat.mod_eq_of_lt (b := n + 1) (by omega), ←add_assoc, Nat.mod_eq_of_lt]
    have := offDiag_def._proof_3 i j
    rwa [← Nat.div_lt_iff_lt_mul (Nat.zero_lt_of_lt j.isLt)]
  · simp only [Fin.isValue, Fin.mk.injEq]
    congr 1
    rw [add_assoc]
    congr
    conv_lhs =>
      arg 1; arg 1
      change (0 : ℤ) ≤ j.val + 1
      equals True => grind
    conv_lhs =>
      arg 1; arg 2; arg 1; arg 1
      change (j.val + 1: ℤ)
    simp only [↓reduceIte, Fin.val_natCast]
    norm_cast
    rw [Nat.mod_eq_of_lt]
    gcongr
    exact j.isLt
