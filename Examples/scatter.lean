import Xla

def permInv {n : ℕ} :
  Xla.SimpleExpr
    [⟨.int, [n]⟩]
    ⟨.int, [n]⟩ :=
  Soir.Expr.ofFn fun x =>
    Xla.scatter (Xla.ofNat (out := ⟨.int, [n]⟩) 0) (Xla.iota n) x

#eval IO.println (permInv (n := 10)).code
/-
%0 = const int [10] 0;
%1 = iota 10;
%2 = scatter; %0, %1, $0
return %2
-/



example (n : ℕ) (σ : Equiv.Perm (Fin n)) :
    permInv.eval (fun i => σ i) = fun i => (σ.symm i : ℤ) := by
  ext i
  have hn := Nat.ne_zero_of_lt i.isLt
  simp [permInv, reduce_xla, reduce_soir, hn]
  have hex : ∃ j, decide (σ j = i) = true := ⟨σ.symm i, by simp⟩
  have hfind : Fin.find (fun j ↦ decide (σ j = i) = true) hex = σ.symm i := by
    rw [Fin.find_eq_iff]
    refine ⟨by simp, fun j hj hσ => hj.ne ?_⟩
    exact σ.injective ((of_decide_eq_true hσ).trans (Equiv.apply_symm_apply σ i).symm)
  rw [Fin.find?_eq_some_find_of_exists hex, hfind]
