import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic
import Mathlib.Algebra.Ring.Defs
import Mathlib.Data.Nat.ModEq
import Mathlib.GroupTheory.Perm.Cycle.Concrete
import Batteries.Data.Fin.Lemmas
import JaxProof.Curry

namespace Jax

--def Tensor (R : Type) : List ℕ → Type
--  | [] => R
--  | n₀ :: ns => Fin n₀ → Tensor R ns

def Tensor (R : Type) (shape : List ℕ) : Type :=
  Curry (shape.map Fin) R

variable {R : Type}

@[ext]
theorem Tensor.ext {s₀ : ℕ} {s : List ℕ} {A B : Tensor R (s₀ :: s)} : (∀ i, A i = B i) → A = B :=
  fun h => funext h

def Tensor.cast {s s' : List ℕ} (h : s = s') : Tensor R s → Tensor R s' :=
  match s, s' with
  | [], [] => id
  | s₀ :: s₁, s₀' :: s₁' =>
    have h₀ : s₀ = s₀' := by injection h
    have h₁ : s₁ = s₁' := by injection h
    fun x i ↦ cast h₁ (x (i.cast h₀.symm))

--def Tensor.cast_apply {s₀ s₀' : ℕ} {s s' : List ℕ} {x : Tensor R (s₀ :: s)}
--  (h₀ : s₀ = s₀') (h : s = s') (i : Fin s₀) (i' : Fin s₀') :
--    i.val = i'.val → (x i).cast h = (x.cast (List.cons_eq_cons.mpr (.intro h₀ h))) i' := by
--  intro hi
--  cases h₀
--  cases h
--  rfl


@[simp]
def filter_pred {n : ℕ} : List (Fin (n + 1)) → List (Fin n)
  | [] => []
  | Fin.mk 0 _ :: xs => filter_pred xs
  | Fin.mk (n + 1) _ :: xs => (Fin.mk n (by omega)) :: filter_pred xs

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
      | Fin.mk 0 _ :: is => filter is (x i₀)
      | Fin.mk (_ + 1) _ :: _ => fun i ↦ filter _ (x i)
    let xs' := xs.map fun ⟨i, x⟩ ↦ ⟨filter_pred i, filter i x⟩
    einprod s' xs'

instance [Add R] {s : List ℕ} : Add (Tensor R s) := inferInstanceAs (Add (Curry (s.map Fin) R))
instance [Zero R] {s : List ℕ} : Zero (Tensor R s) := inferInstanceAs (Zero (Curry (s.map Fin) R))
instance [AddCommMonoid R] (s : List ℕ) : AddCommMonoid (Tensor R s) :=
  inferInstanceAs (AddCommMonoid (Curry (s.map Fin) R))

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

@[simp]
def Tensor.cumsum [AddCommMonoid R] {s : List ℕ} (x : Tensor R s) : Tensor R s :=
  match s with
  | [] => x
  | [_] => fun i => ∑ j with j ≤ i, x j
  | _ :: _ :: _ => fun i => cumsum (x i)

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
  simp only [Fin.divNat, Fin.mulAdd, ← Fin.val_eq_val]
  have : 0 < m := Nat.zero_lt_of_lt j.isLt
  rw [mul_comm _ m, Nat.mul_add_div this, (Nat.div_eq_zero_iff_lt this).mpr j.isLt, add_zero]

@[simp]
theorem _root_.Fin.modNat_mulAdd {n m : ℕ} (i : Fin n) (j : Fin m) : (i.mulAdd j).modNat = j := by
  simp only [Fin.modNat, Fin.mulAdd, Nat.mul_add_mod_self_right, ← Fin.val_eq_val]
  exact Nat.mod_eq_of_lt j.isLt

@[simp]
theorem _root_.Fin.mulAdd_divNat_modNat {n m : ℕ} (i : Fin (n * m)) :
    i.divNat.mulAdd i.modNat = i := by
  simp only [Fin.mulAdd, Fin.coe_divNat, Fin.coe_modNat, ← Fin.val_eq_val]
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
    simp only [List.prod_nil, flatten, unflatten, Fin.isValue]
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

def Tensor.preBroadcast (s : List (ℕ × Bool)) : List ℕ :=
  (s.filter Prod.snd).map Prod.fst

@[simp]
theorem Tensor.preBroadcast_append (s s' : List (ℕ × Bool)) :
    Tensor.preBroadcast (s ++ s') = Tensor.preBroadcast s ++ Tensor.preBroadcast s' := by
  simp [preBroadcast]

def Tensor.broadcast (s : List (ℕ × Bool)) :
    Tensor R (preBroadcast s) → Tensor R (s.map Prod.fst) :=
  match s with
  | [] => id
  | (_, true) :: s => fun x i ↦ broadcast s (x i)
  | (_, false) :: s => fun x _ ↦ x.broadcast s
    

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
  | _ :: _ => fun x i₀ i₁ ↦ uncurry (x i₁) i₀

def Tensor.uncurry' {s s' : List ℕ} : Tensor (Tensor R s') s → Tensor R (s ++ s') :=
  match s with
  | [] => id
  | _ :: _ => fun x i => uncurry' (x i)

def Tensor.map₃ {s : List ℕ} {α β γ μ : Type} (f : α → β → γ → μ) :
    Tensor α s → Tensor β s → Tensor γ s → Tensor μ s :=
  match s with
  | [] => f
  | _ :: _ => fun x y z i ↦ map₃ f (x i) (y i) (z i)

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

def Tensor.map {s : List ℕ} {R R' : Type} (f : R → R') : Tensor R s → Tensor R' s :=
  match s with
  | [] => f
  | _ :: _ => fun x i₀ ↦ (x i₀).map f

def Tensor.batchGet_to_batchGetInt {s s' : List ℕ} (hs : ∀ l ∈ s, l ≠ 0) :
    batchGetType R s' s → batchGetIntType R s' s.length :=
  match s with
  | [] => id
  | s₀ :: s => fun x i₀ ↦
    have : NeZero s₀ := ⟨by simp [hs]⟩
    let i₀' : Tensor (Fin s₀) s' := i₀.map Fin.intCast
    batchGet_to_batchGetInt
      (by simp only [List.mem_cons, ne_eq, forall_eq_or_imp] at hs; exact hs.2)
      (x i₀')

def ValidIdx (s : List ℕ) : Type := ∀ i : Fin s.length, Fin (s.get i)

def Tensor.get {s : List ℕ} : Tensor R s → ValidIdx s → R :=
  match s with
  | [] => fun x _ ↦ x
  | n₀ :: ns => fun x i ↦ Tensor.get (x (i ⟨0, by simp⟩)) fun j ↦ i j.succ

def Tensor.of {s : List ℕ} : (ValidIdx s → R) → Tensor R s :=
  match s with
  | [] => fun x ↦ x (fun i ↦ nomatch i)
  | n₀ :: ns => fun x i₀ ↦
    let x' : ValidIdx ns → R := fun is ↦
      let i : ValidIdx (n₀ :: ns) := fun r ↦
        match r with
        | Fin.mk 0 _ => i₀
        | Fin.mk (r + 1) h => is <| Fin.mk r <| by simp at h; omega
      x i
    Tensor.of x'

@[simp]
theorem Tensor.get_of {s : List ℕ} {x : ValidIdx s → R} : (Tensor.of x).get = x := by
  ext i
  induction s with
  | nil =>
    simp only [get, of, List.length_nil, List.get_eq_getElem]
    congr
    apply funext
    intro r
    nomatch r
  | cons s₀ s ih =>
    simp only [get, of, List.length_cons, List.get_eq_getElem, Fin.zero_eta, ih, Fin.succ_mk]
    congr
    conv_lhs =>
      intro r
      equals i r =>
        match r with
        | 0 => rfl
        | .mk (_ + 1) _ => rfl

@[simp]
theorem Tensor.of_get {s : List ℕ} {x : Tensor R s} : Tensor.of x.get = x := by
  induction s with
  | nil => rfl
  | cons s₀ s ih =>
    simp only [of, get, List.length_cons, List.get_eq_getElem]
    conv_lhs =>
      intro i₀; arg 1
      equals (x i₀).get =>
        ext is
        congr
    simp [ih]

def Tensor.transpose {s : List ℕ} (σ : Equiv.Perm (Fin s.length)) :
    Tensor R s → Tensor R (List.ofFn fun i ↦ s.get (σ i)) :=
  fun x ↦ Tensor.of fun i ↦ x.get fun μ ↦ 
    let j := i <| (σ.symm μ).cast <| by simp
    j.cast <| by simp

@[simps]
instance [Div R] (s : List ℕ) : Div (Tensor R s) where div := Tensor.map₂ (· / ·)

example (n m l : ℕ) (A : Matrix (Fin n) (Fin m) ℝ) (B : Matrix (Fin m) (Fin l) ℝ) :
    Tensor.einsum [m, n, l] [⟨[#1, #0], A⟩, ⟨[#0, #2], B⟩] 1 = A * B := by
  simp only [List.drop_succ_cons, List.drop_zero, List.length_cons, List.length_nil, Nat.reduceAdd,
    Fin.mk_one, Fin.isValue, Fin.zero_eta, Fin.reduceFinMk] 
  ext i j
  simp only [Tensor.einsum, Tensor.sumN, Tensor.sumFirst, Tensor.einprod, List.length_nil,
    List.map_nil, List.length_cons, Nat.reduceAdd, Fin.isValue, List.map_cons, filter_pred,
    Fin.zero_eta, Tensor.einprod.filter, List.get_eq_getElem, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
    List.getElem_cons_zero, Fin.mk_one, Nat.reduceMod, List.getElem_cons_succ, List.prod_cons,
    List.prod_nil, mul_one, List.tail_cons, Matrix.mul_apply]
  conv_lhs =>
    change (∑ k, fun i j ↦ A i k * B k j) i j
  simp [Finset.sum_apply]

example (i : Fin 2) (j : Fin 3) (k : Fin 4) (x : Tensor R [2, 4]) :
    let y : Tensor R [2,3,4] := x.broadcast [(2,true),(3,false),(4,true)]
    y i j k = x i k :=
  rfl

example (x : Tensor R [2, 3]) (i : Tensor (Fin 2) [4, 5]) (j : Tensor (Fin 3) [4, 5])
  (a : Fin 4) (b : Fin 5) : x.batchGet i j a b = x (i a b) (j a b) := rfl

example (n₁ n₂ : ℕ) (x : Tensor R [n₁, n₂]) (i : Fin n₁) (j : Fin n₂) :
    x.transpose [0,1].formPerm j i = x i j := rfl

noncomputable def softmax {n₁ n₂ : ℕ} (x : Tensor ℝ [n₁, n₂]) : Tensor ℝ [n₁, n₂] :=
  let denom := Tensor.einsum [n₂, n₁] [⟨[#1, #0], x⟩] 1
  let denom' : Tensor ℝ [n₁, n₂] := denom.broadcast [(n₁, true), (n₂, false)]
  x / denom'

example (n₁ n₂ : ℕ) (x : Tensor ℝ [n₁, n₂]) (i : Fin n₁) (j : Fin n₂) :
    softmax x i j = x i j / ∑ k, x i k := by
  simp only [softmax, Tensor.broadcast, Tensor.einsum, Tensor.sumN, Tensor.sumFirst, Tensor.einprod,
    List.length_nil, List.map_nil, List.length_cons, Nat.reduceAdd, Fin.mk_one, Fin.isValue,
    Fin.zero_eta, List.map_cons, filter_pred, Tensor.einprod.filter, List.get_eq_getElem,
    Fin.coe_ofNat_eq_mod, Nat.zero_mod, List.getElem_cons_zero, List.prod_cons, List.prod_nil,
    mul_one, List.tail_cons, id_eq, div_def, Tensor.map₂]
  apply congrArg
  conv_lhs =>
    change (∑ j, fun i ↦ x i j) i
  rw [Finset.sum_apply]

end Jax
