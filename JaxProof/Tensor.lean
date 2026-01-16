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

@[simp]
theorem ValidIdx.unflatten_flatten (s : List ℕ) (x : ValidIdx s) :
    ValidIdx.unflatten s x.flatten = x := by
  induction s with
  | nil => simp only [ValidIdx, List.length_nil, List.get_eq_getElem]; ext i; nomatch i
  | cons s₀ s₁ ih =>
    have : NeZero (s₀ :: s₁).length := ⟨by simp⟩
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

@[simp]
theorem ValidIdx.flatten_unflatten (s : List ℕ) (x : Fin s.prod) :
    (ValidIdx.unflatten s x).flatten = x := by
  induction s with
  | nil =>
    simp[flatten]
    rw[←Fin.val_eq_val, Fin.val_zero]
    omega
  | cons s₀ s₁ ih =>
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

def Tensor (R : Type) : List ℕ → Type
  | [] => R
  | n₀ :: ns => Fin n₀ → Tensor R ns

variable {R : Type}

@[ext]
theorem Tensor.ext {s₀ : ℕ} {s : List ℕ} {A B : Tensor R (s₀ :: s)} : (∀ i, A i = B i) → A = B :=
  fun h => funext h

def Tensor.get {s : List ℕ} : Tensor R s → Jax.ValidIdx s → R :=
  match s with
  | [] => fun x _ ↦ x
  | n₀ :: ns => fun x i ↦ Tensor.get (x (i ⟨0, by simp⟩)) fun j ↦ i j.succ

def Tensor.of {s : List ℕ} : (Jax.ValidIdx s → R) → Tensor R s :=
  match s with
  | [] => fun x ↦ x (fun i ↦ nomatch i)
  | n₀ :: ns => fun x i₀ ↦
    let x' : Jax.ValidIdx ns → R := fun is ↦
      let i : Jax.ValidIdx (n₀ :: ns) := fun r ↦
        if h : r = ⟨0, by simp⟩ then
          i₀.cast <| by simp [h]
        else 
          (is (r.pred h)).cast <| by
            nth_rw 2 [← Fin.succ_pred r h]
            rw [List.get_cons_succ']
      x i
    Tensor.of x'

@[simp]
theorem Tensor.of_get {s : List ℕ} (x : Tensor R s) : Tensor.of x.get = x := by
  induction s with
  | nil =>
    simp [of, get]
  | cons n₀ ns ih =>
    simp [of, get]
    conv_lhs =>
      intro i₀; arg 1
      change (x i₀).get
    simp [ih]

@[simp]
theorem Tensor.get_of {s : List ℕ} (x : Jax.ValidIdx s → R) : (Tensor.of x).get = x := by
  induction s with
  | nil =>
    simp [of, get]; ext i; congr; apply funext; intro j; nomatch j
  | cons n₀ ns ih =>
    simp [of, get, ih]
    ext i
    congr
    apply funext
    intro r
    split_ifs with h
    · congr
      · simp [h]
      · exact proof_irrel_heq _ _
      · exact h.symm
    congr 1
    · simp
      have : r.val ≠ 0 := fun hc ↦ h <| by simpa using hc
      conv_rhs =>
        arg 2
        equals r.val - 1 + 1 =>
          refine (Nat.sub_add_cancel ?_).symm
          omega
      simp
    · exact proof_irrel_heq _ _
    congr
    exact Fin.succ_pred _ _

@[simp]
def filter_pred {n : ℕ} : List (Fin (n + 1)) → List (Fin n)
  | [] => []
  | x :: xs => 
    if h : x = 0 then
      filter_pred xs
    else
      (x.pred h) :: filter_pred xs

def Tensor.einprod [Mul R] [One R] (s : List ℕ)
  (xs : List ((i : List (Fin s.length)) × Tensor R (i.map s.get))) : Tensor R s :=
  match s with
  | [] => 
    let xs' : List R := xs.map fun ⟨i, x⟩ ↦ match i with | [] => x
    xs'.prod
  | s₀ :: s' => fun i₀ ↦
    let rec filter (i : List (Fin (s'.length + 1))) (x : Tensor R (i.map (s₀ :: s').get)) :
      Tensor R ((filter_pred i).map s'.get) :=
      match i with
      | [] => x
      | i₁ :: is => by
        unfold filter_pred
        split_ifs with h
        · exact filter is (x (i₀.cast (by simp[h])))
        · exact fun i' ↦ filter is <| x <| i'.cast <| by
            have ⟨i₁', h'⟩ := Fin.eq_succ_of_ne_zero h
            simp [h']
    let xs' := xs.map fun ⟨i, x⟩ ↦ ⟨filter_pred i, filter i x⟩
    einprod s' xs'

def Tensor.add [Add R] {s : List ℕ} (A B : Tensor R s) : Tensor R s :=
  match s with
  | [] => @id R A + @id R B
  | _ :: _ => fun i ↦ add (A i) (B i)

def Tensor.zero [Zero R] {s : List ℕ} : Tensor R s :=
  match s with
  | [] => (0 : R)
  | _ :: _ => fun _ ↦ zero

instance [Add R] {s : List ℕ} : Add (Tensor R s) := ⟨Tensor.add⟩
instance [Zero R] {s : List ℕ} : Zero (Tensor R s) := ⟨Tensor.zero⟩

@[simp]
theorem Tensor.add_apply [Add R] (s₀ : ℕ) {s : List ℕ} {x y : Tensor R (s₀ :: s)} {i : Fin s₀} :
    (x + y) i = x i + y i := by
  simp [HAdd.hAdd, Add.add, add]

@[simp]
theorem Tensor.zero_apply [Zero R] (s₀ : ℕ) {s : List ℕ} {i : Fin s₀} :
    (0 : Tensor R (s₀ :: s)) i = (0 : Tensor R s) := by
  simp[OfNat.ofNat, Zero.zero, zero]

theorem Tensor.zero_add [AddCommMonoid R] {s : List ℕ} : ∀ x : Tensor R s, 0 + x = x := by
  induction s with
  | nil => simp
  | cons s₀ s ih =>
    intro x
    ext i
    simp [ih]

theorem Tensor.add_zero [AddCommMonoid R] {s : List ℕ} : ∀ x : Tensor R s, x + 0 = x := by
  induction s with
  | nil => simp
  | cons s₀ s ih =>
    intro x
    ext i
    simp [ih]

theorem Tensor.add_assoc [AddCommMonoid R] {s : List ℕ} (x y z : Tensor R s) :
    x + y + z = x + (y + z) := by
  induction s with
  | nil => simp [_root_.add_assoc]
  | cons s₀ s ih =>
    ext i
    simp [ih]

theorem Tensor.add_comm [AddCommMonoid R] {s : List ℕ} (x y : Tensor R s) :
    x + y = y + x := by
  induction s with
  | nil => simp [_root_.add_comm]
  | cons s₀ s ih =>
    ext i
    simp [ih]

instance [AddCommMonoid R] (s : List ℕ) : AddCommMonoid (Tensor R s) where
  nsmul n x := Nat.repeat (· + x) n 0
  zero_add := Tensor.zero_add
  add_zero := Tensor.add_zero
  add_assoc := Tensor.add_assoc
  add_comm := Tensor.add_comm

@[simp]
def Tensor.sumFirst [AddCommMonoid R] {s : List ℕ} (x : Tensor R s) : Tensor R s.tail :=
  match s with
  | [] => x 
  | s₀ :: _ => ∑ i : Fin s₀, x i

@[simp]
def Tensor.sumN [AddCommMonoid R] {s : List ℕ} (n : ℕ) (x : Tensor R s) : Tensor R (s.drop n) :=
  match n, s with
  | 0, _ => x
  | _ + 1, [] => x
  | n + 1, _ :: _ => sumN n x.sumFirst

def Tensor.einsum [AddCommMonoid R] [Mul R] [One R] (s : List ℕ)
  (xs : List ((i : List (Fin s.length)) × Tensor R (i.map s.get))) (nsum : ℕ) :
    Tensor R (s.drop nsum) :=
  (einprod s xs).sumN nsum

def Tensor.reshape (s s' : List ℕ) (h : s.prod = s'.prod) (x : Tensor R s) : Tensor R s' :=
  Tensor.of fun i ↦ x.get (ValidIdx.unflatten s (i.flatten.cast h.symm))

--def Tensor.flatten {s : List ℕ} : Tensor R s → Fin s.prod → R :=
--  fun x i ↦ x.get (ValidIdx.unflatten s i)

def Tensor.flatten {s : List ℕ} : Tensor R s → Fin s.prod → R :=
  match s with
  | [] => fun x _ ↦ x
  | _ :: _ => fun x i ↦ flatten (x i.divNat) i.modNat


def Tensor.unflatten (s : List ℕ) : (Fin s.prod → R) → Tensor R s :=
  fun x ↦ Tensor.of fun i ↦ x i.flatten

attribute [simp] Tensor.einprod.filter

macro "#" noWs n:num : term => `(⟨$n, by simp +decide⟩)

example (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) :
    Tensor.einsum [n, n, n] [⟨[#1, #0], A⟩, ⟨[#0, #2], B⟩] 1 = A * B := by
  simp 
  ext i j
  simp [Tensor.einsum, Tensor.einprod, show (2 : Fin 3) ≠ 0 by decide, Matrix.mul_apply]
  have h₁ := Finset.sum_apply i Finset.univ fun j i k ↦ A i j * B j k
  conv_lhs =>
    fun
    equals ∑ j, fun k ↦ A i j * B j k =>
      exact h₁
  simp [Finset.sum_apply]

--class TensorLike (dtype : Type) where
--  protected tensor : List ℕ → dtype → Type

end Jax
