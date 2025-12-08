import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Algebra.Ring.Defs
import Mathlib.Data.Nat.ModEq
import Batteries.Data.Fin.Lemmas

namespace Jax

def curryType (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => α → curryType α n


def ValidShape : Type := Subtype fun (s : List ℕ) ↦ ∀ n ∈ s, n ≠ 0

def ValidIdx (s : ValidShape) : Type := ∀ i : Fin s.val.length, Fin (s.val.get i)

instance (s : ValidShape) : Fintype (ValidIdx s) :=
  inferInstanceAs (Fintype (∀ i : Fin s.val.length, Fin (s.val.get i)))

instance (s : ValidShape) : DecidableEq (ValidIdx s) :=
  inferInstanceAs (DecidableEq (∀ i : Fin s.val.length, Fin (s.val.get i)))

def ValidShape.size (s : ValidShape) : ℕ := s.val.prod

@[simp]
def subshape (s : ValidShape) (l : List (Fin s.val.length)) : ValidShape :=
  Subtype.mk (l.map s.val.get) <| by
    intro n hn
    rw [List.mem_map] at hn
    obtain ⟨i, hi⟩ := hn
    rw [← hi.2]
    exact s.prop (s.val.get i) (List.get_mem _ _)

def flattenIdx {s : ValidShape} (idx : ValidIdx s) : Fin s.size :=
  let ⟨sval, sprop⟩ := s
  match sval with
  | [] => ⟨0, by simp [ValidShape.size]⟩
  | s₀ :: s₁ =>
    let s' : ValidShape := ⟨s₁, by simp at sprop; exact sprop.2⟩
    let idx' : ValidIdx s' := fun i ↦ idx i.succ
    have i' := flattenIdx idx'
    let i₀ := idx ⟨0, by simp⟩
    Fin.mk (i₀ + s₀ * i') <| by
      simp [ValidShape.size]
      rw [← Nat.succ_le_iff, Nat.succ_eq_add_one]
      calc
        _ = (i₀ + 1) + s₀ * i' := by ac_nf
        _ ≤ s₀ * (i' + 1) := by rw[add_comm i'.val 1, Nat.mul_add, mul_one]; grind
        _ ≤ _ := by gcongr; rw[Nat.succ_le_iff]; exact i'.isLt
termination_by s.val

def unflattenIdx (s : ValidShape) (idx : Fin s.size) : ValidIdx s :=
  let ⟨sval, sprop⟩ := s
  match sval with
  | [] => fun i ↦ nomatch i
  | s₀ :: s₁ => fun i ↦
    let idx' : Fin (s₁.prod * s₀) := idx.cast <| by simp [ValidShape.size]; rw[mul_comm]
    if h : i = ⟨0, _⟩ then
      idx'.modNat.cast <| by simp [h]
    else
      (unflattenIdx ⟨s₁, by grind⟩ idx'.divNat (i.pred h)).cast <| by
        simp [List.getElem_cons]
        intro hc
        exact (h hc).elim
termination_by s.val

def flattenEuiv (s : ValidShape) : ValidIdx s ≃ Fin s.size where
  toFun := flattenIdx
  invFun := unflattenIdx s
  left_inv := by
    let ⟨sval, sprop⟩ := s
    induction sval with
    | nil => intro x; simp[ValidIdx]; ext i; nomatch i
    | cons s₀ s₁ ih =>
      have : NeZero (s₀ :: s₁).length := ⟨by simp⟩
      intro x
      simp[flattenIdx, unflattenIdx, ValidIdx]
      ext i
      split_ifs with h
      · simp only [Fin.coe_cast, Fin.coe_modNat, Nat.add_mul_mod_self_left]
        rw[h]
        exact Nat.mod_eq_of_lt (x 0).isLt
      simp only [List.mem_cons, ne_eq, forall_eq_or_imp] at sprop
      specialize ih sprop.2 (fun i ↦ x i.succ)
      simp only [Fin.divNat, Fin.coe_cast]
      conv =>
        lhs; arg 1; arg 2; arg 1
        rw[Nat.add_div_of_dvd_left (by simp), 
          show (x 0).val / s₀ = 0 from Nat.div_eq_zero_iff.mpr (Or.inr (x 0).isLt), zero_add,
          mul_comm, Nat.mul_div_cancel _ (by omega)]
      simp only [Fin.eta, ih]
      let g (j : Fin (s₁.length + 1)) : ℕ := x i
      have : x (i.pred h).succ = g (i.pred h).succ := by congr; all_goals exact Fin.succ_pred i h
      rw[this]
  right_inv := by
    let ⟨sval, sprop⟩ := s
    induction sval with
    | nil =>
      intro x
      simp[flattenIdx]
      simp[ValidShape.size] at x
      rw[←Fin.val_eq_val, Fin.val_zero]
      omega
    | cons s₀ s₁ ih =>
      intro x
      simp[unflattenIdx, flattenIdx]
      rw [← Fin.val_eq_val]
      push_cast
      conv_rhs =>
        rw [← Nat.mod_add_div x.val s₀]
      congr
      simp only [List.mem_cons, ne_eq, forall_eq_or_imp] at sprop
      specialize ih sprop.2
        (x.cast (by simp[ValidShape.size, mul_comm]) : Fin (s₁.prod * s₀)).divNat
      simp [Fin.divNat, ← Fin.val_eq_val] at ih
      rw[← ih]
      congr

end Jax
