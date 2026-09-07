import Xla

lemma sq_sub_one (n : ℕ) : n * n - 1 = (n - 1) * (n + 1) := by
  by_cases h : n = 0
  · rw [h]
  rw [Nat.sub_mul, mul_add, mul_one, one_mul, ← Nat.sub_sub, Nat.add_sub_cancel]

def offDiag {n : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n, n]⟩]
    ⟨.float, [n - 1, n]⟩ :=
  Soir.Expr.ofFn fun x =>
    let x := Xla.flatten x;
    let x := Xla.gather x (Xla.iota (n * n - 1));
    let x := Xla.cast (show [n * n - 1] = [(n - 1) * ((n + 1) * 1)] by congr; simp[sq_sub_one]) x;
    let x := Xla.unflatten [n - 1, n + 1] x;
    let i := Xla.broadcast [⟨n - 1, true⟩, ⟨n, false⟩] (Xla.iota (n - 1));
    let j := Xla.add (Xla.broadcast [⟨n - 1, false⟩, ⟨n, true⟩] (Xla.iota n)) (Xla.ofNat 1);
    let x := Xla.gather x (i.append j);
    x

#eval IO.println (offDiag (n := 8)).code

def offDiag_def {n : ℕ} (x : Xla.Tensor ℝ [n, n]) : Xla.Tensor ℝ [n - 1, n] :=
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

theorem offDiag_eq_def {n : ℕ} : (offDiag (n := n)).eval = offDiag_def := by
  funext x
  ext i j
  simp [offDiag, offDiag_def, reduce_xla, reduce_soir, reduce_tensor]
  have hn : 2 ≤ n := by have := i.isLt; omega
  have hnn : NeZero (n * (n * 1)) :=
    ⟨by rw [Nat.mul_one]; exact Nat.mul_ne_zero_iff.mpr ⟨by omega, by omega⟩⟩
  have hlt : i.val * (n + 1) + (j.val + 1) < n * n := by
    have hi := Nat.succ_le_iff.mpr i.isLt
    have hj := Nat.succ_le_iff.mpr j.isLt
    calc
      (i.val * (n + 1) + j.val + 1).succ = i.val.succ * (n + 1) + j.val.succ - n := by grind
      _ ≤ (n - 1) * (n + 1) + j.val.succ - n := by gcongr
      _ ≤ (n - 1) * (n + 1) + n - n := by gcongr
      _ ≤ n * n := by rw [← sq_sub_one, Nat.add_sub_cancel]; grind
  rw [if_neg (by omega : n - 1 ≠ 0)]
  split_ifs with h
  · have := i.isLt; omega
  · have hz : (↑i.val * (↑n + 1) + (↑j.val + 1) : ℤ) =
        ((i.val * (n + 1) + (j.val + 1) : ℕ) : ℤ) := by push_cast; ring
    congr 1
    · ext
      rw [Fin.coe_divNat, hz, Fin.val_intCast_natCast, Nat.mul_one, Nat.mod_eq_of_lt hlt]
      rfl
    · ext
      rw [Fin.coe_divNat, Nat.div_one, Fin.coe_modNat, hz, Fin.val_intCast_natCast, Nat.mul_one,
        Nat.mod_eq_of_lt hlt]
      rfl

