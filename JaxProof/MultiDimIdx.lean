import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic
import Mathlib.Algebra.Ring.Defs
import Mathlib.Data.Nat.ModEq
import Batteries.Data.Fin.Lemmas

namespace Jax

def curryType (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => α → curryType α n


def ValidIdx (s : List ℕ) : Type := ∀ i : Fin s.length, Fin (s.get i)

instance (s : List ℕ) : Fintype (ValidIdx s) :=
  inferInstanceAs (Fintype (∀ i : Fin s.length, Fin (s.get i)))

instance (s : List ℕ) : DecidableEq (ValidIdx s) :=
  inferInstanceAs (DecidableEq (∀ i : Fin s.length, Fin (s.get i)))

def ValidIdx.flatten {s : List ℕ} (idx : ValidIdx s) : Fin s.prod :=
  match s with
  | [] => ⟨0, by simp⟩
  | s₀ :: s₁ =>
    let idx' : ValidIdx s₁ := fun i ↦ idx i.succ
    have i' := idx'.flatten
    let i₀ := idx ⟨0, by simp⟩
    Fin.mk (i₀ * s₁.prod + i') <| by
      simp
      rw [← Nat.succ_le_iff, Nat.succ_eq_add_one]
      calc
        _ ≤ (i₀.val + 1) * s₁.prod := by
          rw [add_mul, one_mul, add_assoc]
          gcongr
          exact Nat.succ_le_of_lt i'.isLt
        _ ≤ _ := by gcongr; exact Nat.succ_le_of_lt i₀.isLt

def ValidIdx.unflatten (s : List ℕ) (idx : Fin s.prod) : ValidIdx s :=
  match s with
  | [] => fun i ↦ nomatch i
  | s₀ :: s₁ => fun i ↦
    if h : i = ⟨0, _⟩ then
      (idx : Fin (s₀ * s₁.prod)).divNat.cast <| by simp[h]
    else
      (unflatten s₁ idx.modNat (i.pred h)).cast <| by
        simp [List.getElem_cons]
        intro hc
        exact (h hc).elim

def ValidIdx.equiv (s : List ℕ) : ValidIdx s ≃ Fin s.prod where
  toFun := flatten
  invFun := unflatten s
  left_inv := by
    induction s with
    | nil => intro x; simp[ValidIdx]; ext i; nomatch i
    | cons s₀ s₁ ih =>
      have : NeZero (s₀ :: s₁).length := ⟨by simp⟩
      intro x
      simp[flatten, unflatten, ValidIdx]
      have hs₁ : 0 < s₁.prod := by
        by_contra
        have hc : s₁.prod = 0 := by omega
        rw [List.prod_eq_zero_iff, List.mem_iff_get] at hc
        obtain ⟨i, hi⟩ := hc
        have := (x i.succ).isLt
        simp only [List.get_cons_succ', hi] at this
        omega
      ext i
      split_ifs with h
      · simp
        set x' := flatten fun i ↦ x i.succ
        conv_lhs =>
          arg 2
          change s₁.prod
        rw [h, Nat.add_div_of_dvd_right]
        · rw [Nat.mul_div_left _ hs₁, (Nat.div_eq_zero_iff_lt hs₁).mpr x'.isLt, add_zero]
        · exact Nat.dvd_mul_left _ _
      simp only [Fin.modNat, Fin.coe_cast]
      specialize ih (fun i ↦ x i.succ)
      have : 0 < s₀ := by
        have := (x 0).isLt
        simp only [List.get_cons_zero] at this
        exact Nat.zero_lt_of_lt this
      conv =>
        lhs; arg 1; arg 2; arg 1
        conv =>
          arg 2; change s₁.prod
        rw [Nat.add_mod]
        rw [Nat.mul_mod_left, zero_add, Nat.mod_mod,
          Nat.mod_eq_of_lt (flatten fun i ↦ x i.succ).isLt]
      simp only [Fin.eta, ih]
      let g (j : Fin (s₁.length + 1)) : ℕ := x i
      have : x (i.pred h).succ = g (i.pred h).succ := by congr; all_goals exact Fin.succ_pred i h
      rw[this]
  right_inv := by
    induction s with
    | nil =>
      intro x
      simp[flatten]
      rw[←Fin.val_eq_val, Fin.val_zero]
      omega
    | cons s₀ s₁ ih =>
      intro x
      simp[unflatten, flatten]
      rw [← Fin.val_eq_val]
      push_cast
      conv_rhs =>
        rw [← Nat.mod_add_div' x.val s₁.prod]
      conv_lhs =>
        arg 1; arg 1; arg 2; change s₁.prod
      nth_rw 1 [add_comm]
      congr
      specialize ih x.modNat
      simp [← Fin.val_eq_val] at ih
      rw[← ih]
      congr

end Jax
