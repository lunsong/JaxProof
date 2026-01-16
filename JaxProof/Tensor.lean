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

def Tensor (R : Type) : List ℕ → Type
  | [] => R
  | n₀ :: ns => Fin n₀ → Tensor R ns

variable {R : Type}

@[ext]
theorem Tensor.ext {s₀ : ℕ} {s : List ℕ} {A B : Tensor R (s₀ :: s)} : (∀ i, A i = B i) → A = B :=
  fun h => funext h

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

def Tensor.flatten {s : List ℕ} : Tensor R s → Fin s.prod → R :=
  match s with
  | [] => fun x _ ↦ x
  | _ :: _ => fun x i ↦ flatten (x i.divNat) i.modNat

macro "#" noWs n:num : term => `(⟨$n, by simp +decide⟩)

def _root_.Fin.mulAdd {n m : ℕ} (i : Fin n) (j : Fin m) : Fin (n * m) :=
  Fin.mk (i.val * m + j) <| by
    rw [← Nat.add_one_le_iff, add_assoc]
    trans (i.val + 1) * m
    · rw [add_mul, one_mul]
      gcongr
      rw [Nat.add_one_le_iff]
      exact j.isLt
    · gcongr
      exact i.isLt

@[simp]
theorem _root_.Fin.divNat_mulAdd {n m : ℕ} (i : Fin n) (j : Fin m) : (i.mulAdd j).divNat = i := by
  simp [Fin.mulAdd, Fin.divNat, ← Fin.val_eq_val]
  have : 0 < m := Nat.zero_lt_of_lt j.isLt
  rw [mul_comm _ m, Nat.mul_add_div this, (Nat.div_eq_zero_iff_lt this).mpr j.isLt, add_zero]

@[simp]
theorem _root_.Fin.modNat_mulAdd {n m : ℕ} (i : Fin n) (j : Fin m) : (i.mulAdd j).modNat = j := by
  simp [Fin.mulAdd, Fin.modNat, ← Fin.val_eq_val]
  exact Nat.mod_eq_of_lt j.isLt

@[simp]
theorem _root_.Fin.mulAdd_divNat_modNat {n m : ℕ} (i : Fin (n * m)) :
    i.divNat.mulAdd i.modNat = i := by
  simp [Fin.mulAdd, ← Fin.val_eq_val]
  exact Nat.div_add_mod' i.val m

def Tensor.unflatten (s : List ℕ) : (Fin s.prod → R) → Tensor R s :=
  match s with
  | [] => fun x ↦ x (0 : Fin 1)
  | _ :: s => fun x i ↦ Tensor.unflatten s fun j ↦ x (i.mulAdd j)

@[simp]
theorem Tensor.flatten_unflatten (s : List ℕ) (x : Fin s.prod → R) :
    (Tensor.unflatten s x).flatten = x := by
  induction s with
  | nil =>
    simp [unflatten, flatten]
    ext i
    fin_cases i
    simp
  | cons s₀ s ih =>
    simp [unflatten, flatten, ih]

@[simp]
theorem Tensor.unflatten_flatten {s : List ℕ} (x : Tensor R s) :
    Tensor.unflatten s x.flatten = x := by
  induction s with
  | nil =>
    simp [unflatten, flatten]
  | cons s₀ s ih =>
    simp [unflatten, flatten, ih]

attribute [simp] Tensor.einprod.filter

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
