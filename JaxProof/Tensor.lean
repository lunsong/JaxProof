import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic
import Mathlib.Algebra.Ring.Defs
import Mathlib.Data.Nat.ModEq
import Mathlib.GroupTheory.Perm.Cycle.Concrete
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

def Tensor.const {s : List ℕ} (x : R) : Tensor R s :=
  match s with
  | [] => x
  | _ :: _ => fun _ ↦ const x

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

--class TensorLike (dtype : Type) where
--  protected tensor : List ℕ → dtype → Type

instance (α : Type) (x₀ : α) (xs : List α) : NeZero (x₀ :: xs).length :=
  NeZero.mk <| by simp

def Tensor.preBroadcast : List (ℕ × Bool) → List ℕ
  | [] => []
  | (s₀, isBroadcast) :: xs =>
    let xs' := preBroadcast xs
    if isBroadcast then xs' else s₀ :: xs'


def Tensor.broadcast (s : List (ℕ × Bool)) :
    Tensor R (preBroadcast s) → Tensor R (s.map Prod.fst) :=
  match s with
  | [] => id
  | (s₀, isBroadcast) :: s => by
    unfold preBroadcast
    split_ifs with h
    · exact fun x _ ↦ x.broadcast s
    · exact fun x i₀ ↦ (x i₀).broadcast s

def Tensor.batchGetType (R : Type) (s' : List ℕ) : List ℕ → Type
  | [] => Tensor R s' 
  | s₀ :: s => Tensor (Fin s₀) s' → Tensor.batchGetType R s' s

def Tensor.fill {s : List ℕ} (x : R) : Tensor R s :=
  match s with
  | [] => x
  | _ :: _ => fun _ ↦ Tensor.fill x

def Tensor.curry {s₀ : ℕ} {s : List ℕ} : Tensor R (s₀ :: s) → Tensor (Tensor R [s₀]) s :=
  match s with
  | [] => id
  | _ :: _ => fun x i₁ ↦ curry fun i₀ ↦ x i₀ i₁

def Tensor.curry' {s s' : List ℕ} : Tensor R (s ++ s') → Tensor (Tensor R s') s :=
  match s with
  | [] => id
  | _ :: _ => fun x i ↦ Tensor.curry' (x i)

def Tensor.uncurry {s₀ : ℕ} {s : List ℕ} : Tensor (Tensor R [s₀]) s → Tensor R (s₀ :: s) :=
  match s with
  | [] => id
  | _ :: _ => fun x i₀ i₁ ↦ (x i₁).uncurry i₀

def Tensor.map₂ {s : List ℕ} {α β γ : Type} (f : α → β → γ) :
    Tensor α s → Tensor β s → Tensor γ s :=
  match s with
  | [] => f
  | _ :: _ => fun x y i ↦ map₂ f (x i) (y i)

def Tensor.batchGetType.uncurry {s₀ : ℕ} {s s' : List ℕ} :
    batchGetType (Tensor R [s₀]) s' s → Tensor (Fin s₀) s' → batchGetType R s' s :=
  match s with
  | [] => map₂ fun x i ↦ x i
  | _ :: _ => fun x i₀ i₁ ↦ (x i₁).uncurry i₀

def Tensor.batchGet {R : Type} {s s' : List ℕ} : Tensor R s → Tensor.batchGetType R s' s :=
  match s with
  | [] => Tensor.fill
  | _ :: _ => fun x ↦ batchGetType.uncurry x.curry.batchGet

def Tensor.batchGetIntType (R : Type) (s : List ℕ) : ℕ → Type
  | 0 => Tensor R s
  | n + 1 => Tensor ℤ s → Tensor.batchGetIntType R s n

def Tensor.cast {R R' : Type} (f : R → R') {s : List ℕ} : Tensor R s → Tensor R' s :=
  match s with
  | [] => f
  | _ :: _ => fun x i₀ ↦ (x i₀).cast f

def Tensor.batchGet_to_batchGetInt {s s' : List ℕ} (hs : ∀ l ∈ s, l ≠ 0) :
    batchGetType R s' s → batchGetIntType R s' s.length :=
  match s with
  | [] => id
  | s₀ :: s => fun x i₀ ↦
    have : NeZero s₀ := ⟨by simp [hs]⟩
    let i₀' : Tensor (Fin s₀) s' := i₀.cast Fin.intCast
    batchGet_to_batchGetInt (by simp at hs; exact hs.2) (x i₀')

def ValidIdx (s : List ℕ) : Type := ∀ i : Fin s.length, Fin (s.get i)

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

def Tensor.transpose {s : List ℕ} (σ : Equiv.Perm (Fin s.length)) :
    Tensor R s → Tensor R (List.ofFn fun i ↦ s.get (σ i)) :=
  fun x ↦ Tensor.of fun i ↦ x.get fun μ ↦ 
    let j := i <| (σ.symm μ).cast <| by simp
    j.cast <| by simp

def Tensor.map {s : List ℕ} {R R' : Type} (f : R → R') : Tensor R s → Tensor R' s :=
  match s with
  | [] => f
  | _ :: _ => fun x i₀ ↦ (x i₀).map f

@[simps]
instance [Div R] (s : List ℕ) : Div (Tensor R s) where div := Tensor.map₂ (· / ·)

example (n m l : ℕ) (A : Matrix (Fin n) (Fin m) ℝ) (B : Matrix (Fin m) (Fin l) ℝ) :
    Tensor.einsum [m, n, l] [⟨[#1, #0], A⟩, ⟨[#0, #2], B⟩] 1 = A * B := by
  simp 
  ext i j
  simp [Tensor.einsum, Tensor.einprod, show (2 : Fin 3) ≠ 0 by decide, Matrix.mul_apply]
  have h₁ := Finset.sum_apply i Finset.univ fun j i k ↦ A i j * B j k
  conv_lhs =>
    fun
    equals ∑ j, fun k ↦ A i j * B j k =>
      exact h₁
  simp [Finset.sum_apply]

example (i : Fin 2) (j : Fin 3) (k : Fin 4) (x : Tensor R [2, 4]) :
    let y : Tensor R [2,3,4] := x.broadcast [(2,false),(3,true),(4,false)]
    y i j k = x i k :=
  rfl

example (x : Tensor R [2, 3]) (i : Tensor (Fin 2) [4, 5]) (j : Tensor (Fin 3) [4, 5])
  (a : Fin 4) (b : Fin 5) : x.batchGet i j a b = x (i a b) (j a b) := rfl

example (n₁ n₂ : ℕ) (x : Tensor R [n₁, n₂]) (i : Fin n₁) (j : Fin n₂) :
    x.transpose [0,1].formPerm j i = x i j := rfl

noncomputable def softmax {n₁ n₂ : ℕ} (x : Tensor ℝ [n₁, n₂]) : Tensor ℝ [n₁, n₂] :=
  let denom := Tensor.einsum [n₂, n₁] [⟨[#1, #0], x⟩] 1
  let denom' : Tensor ℝ [n₁, n₂] := denom.broadcast [(n₁, false), (n₂, true)]
  x / denom'

example (n₁ n₂ : ℕ) (x : Tensor ℝ [n₁, n₂]) (i : Fin n₁) (j : Fin n₂) :
    softmax x i j = x i j / ∑ k, x i k := by
  simp [softmax, Tensor.map₂, Tensor.broadcast, Tensor.einsum, Tensor.einprod]
  apply congrArg
  conv_lhs =>
    change (∑ j, fun i ↦ x i j) i
  rw [Finset.sum_apply]

end Jax
