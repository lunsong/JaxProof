import JaxProof.Expr
import Mathlib.Data.Finset.Filter
import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Data.Fintype.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace Jax

inductive Array where
  | float : List ℝ → Array
  | int : List ℤ → Array
  | error : Array

open Fin.IntCast

def Array.idx : Array → Array → Array
  | float x, int i =>
    if h : x.length = 0 then
      error
    else
      let : NeZero x.length := ⟨h⟩
      float <| List.ofFn <| fun a ↦ x.get (i.get a)
  | int x, int i =>
    if h : x.length = 0 then
      error
    else
      let : NeZero x.length := ⟨h⟩
      int <| List.ofFn <| fun a ↦ x.get (i.get a)
  | _, _ => error

def Array.setIdx : Array → Array → Array → Array
  | float x, int i, float y =>
    if h₀ : x.length = 0 then error else
    let : NeZero x.length := ⟨h₀⟩
    let i' : List (Fin x.length) := i.map Int.cast
    have h₂ : i'.length = i.length := by simp[i']
    if ¬ i'.Nodup then error else
    if h₁ : y.length = i.length then
      float <| List.ofFn <|
        fun a ↦ match i'.finIdxOf? a with
          | some j => y.get (j.cast (h₂.trans h₁.symm))
          | none => x.get a
    else error
  | int x, int i, int y =>
    if h₀ : x.length = 0 then error else
    let : NeZero x.length := ⟨h₀⟩
    let i' : List (Fin x.length) := i.map Int.cast
    have h₂ : i'.length = i.length := by simp[i']
    if ¬ i'.Nodup then error else
    if h₁ : y.length = i.length then
      int <| List.ofFn <|
        fun a ↦ match i'.finIdxOf? a with
          | some j => y.get (j.cast (h₂.trans h₁.symm))
          | none => x.get a
    else error

    | _, _, _ => error

def Array.add : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      float <| List.ofFn <| fun (i : Fin x.length) => x.get i + y.get (i.cast h)
    else
      error
  | int x, int y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => x.get i + y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.sub : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      float <| List.ofFn <| fun (i : Fin x.length) => x.get i - y.get (i.cast h)
    else
      error
  | int x, int y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => x.get i - y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.mul : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      float <| List.ofFn <| fun (i : Fin x.length) => x.get i * y.get (i.cast h)
    else
      error
  | int x, int y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => x.get i * y.get (i.cast h)
    else
      error
  | _, _ => error

noncomputable def Array.div : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      if 0 ∈ y then
        error
      else
        float <| List.ofFn <| fun (i : Fin x.length) => x.get i / y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.divInt : Array → Array → Array
  | int x, int y =>
    if h : x.length = y.length then
      if 0 ∈ y then
        error
      else
        int <| List.ofFn <| fun (i : Fin x.length) => x.get i / y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.mod : Array → Array → Array
  | int x, int y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => x.get i % y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.toFloat : Array → Array
  | int x => float (x.map Int.cast)
  | _ => error

def Array.rep (n : ℕ) : Array → Array
  | float x => float <| List.ofFn <| fun (i : Fin (x.length * n)) => x.get i.divNat
  | int x => int <| List.ofFn <| fun (i : Fin (x.length * n)) => x.get i.divNat
  | error => error

noncomputable def Array.eq : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => if x.get i = y.get (i.cast h) then 1 else 0
    else
      error
  | int x, int y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => if x.get i = y.get (i.cast h) then 1 else 0
    else
      error
  | _, _ => error
 
noncomputable def Array.lt : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => if x.get i < y.get (i.cast h) then 1 else 0
    else
      error
  | int x, int y =>
    if h : x.length = y.length then
      int <| List.ofFn <| fun (i : Fin x.length) => if x.get i < y.get (i.cast h) then 1 else 0
    else
      error
  | _, _ => error

def Array.select : Array → Array → Array → Array
  | int c, float x, float y =>
    if h : c.length = x.length ∧ c.length = y.length then
      float <| List.ofFn <| fun (i : Fin c.length) =>
        if c.get i = 0 then y.get (i.cast h.2) else x.get (i.cast h.1)
    else
      error
  | int c, int x, int y =>
    if h : c.length = x.length ∧ c.length = y.length then
      int <| List.ofFn <| fun (i : Fin c.length) =>
        if c.get i = 0 then y.get (i.cast h.2) else x.get (i.cast h.1)
    else
      error
  | _, _, _ => error

def addIdx {α : Type} [AddCommMonoid α] (x y : List α) (i : List ℤ) : Option (List α) :=
    if h₀ : x.length = 0 then none else
    let : NeZero x.length := ⟨h₀⟩
    let i' : List (Fin x.length) := i.map Int.cast
    have h₂ : i'.length = i.length := by simp[i']
    if h₁ : y.length = i.length then
      have h := h₂.trans h₁.symm
      some <| .ofFn fun (j : Fin x.length) ↦ ∑ k : Fin i'.length with i'.get k = j, y.get (k.cast h)
    else
      none

nonrec def Array.addIdx : Array → Array → Array → Array
  | float x, int i, float y =>
    match addIdx x y i with
    | some z => .float z
    | none => error
  | int x, int i, int y =>
    match addIdx x y i with
    | some z => .int z
    | none => error
  | _, _, _ => error

namespace Einsum

def multiDimIdx (shape idx : List ℕ) : ℕ × ℕ:=
  (idx.zip shape).foldr (fun a b ↦ (a.1 * b.2 + b.1, a.2 * b.2)) (0, 1)

def valid_idx : List (ℕ × ℕ) → Prop := fun l ↦ ∀ x ∈ l, x.1 < x.2

theorem multiDimIdx_inbound {shape idx : List ℕ} (h : valid_idx (idx.zip shape)) :
    (multiDimIdx shape idx).1 < (multiDimIdx shape idx).2 := by
  revert idx
  induction shape with
  | nil => simp [multiDimIdx]
  | cons H T ih =>
    intro idx
    simp [multiDimIdx]
    match idx with
    | [] => simp
    | H' :: T' =>
      simp
      rw[←multiDimIdx]
      intro h
      simp[valid_idx] at h
      have : valid_idx (T'.zip T) := by simpa[valid_idx] using h.2
      specialize ih this
      generalize (multiDimIdx T T').1 = A at *
      generalize (multiDimIdx T T').2 = B at *
      rw[← Nat.succ_le_iff, Nat.succ_eq_add_one, add_assoc]
      calc
        _ ≤ (H' + 1) * B := by rw[add_mul, one_mul]; gcongr; rwa[Nat.succ_le_iff]
        _ ≤ _ := by gcongr; rw[Nat.succ_le_iff]; exact h.1

theorem multiDimIdx_eq_prod {shape idx : List ℕ} (h : ∀ n ∈ shape, n ≠ 0) :
    (multiDimIdx shape idx).2 ≤ shape.prod := by
  revert idx
  induction shape with
  | nil => simp [multiDimIdx]
  | cons H T ih =>
    simp[multiDimIdx]
    intro idx
    match idx with
    | [] =>
      simp at h ⊢
      have : T.prod ≠ 0 := List.prod_ne_zero <| fun hc ↦ h.2 0 hc rfl
      generalize T.prod = t at *
      have h' : H ≠ 0 := h.1
      by_contra! hc
      have h'' : H * t = 0 := by omega
      simp at h''
      exact h''.elim h' this
    | H' :: T' =>
      simp
      rw[← multiDimIdx]
      simp at h
      specialize @ih h.2 T'
      gcongr

def multiDimIdx' (shape : List ℕ) (idx : (i : Fin shape.length) → Fin (shape.get i))
  (hshape : ∀ n ∈ shape, n ≠ 0) : Fin shape.prod :=
  Fin.mk (multiDimIdx shape (List.ofFn (idx ·))).1 <| by
    refine lt_of_lt_of_le (multiDimIdx_inbound ?_) (multiDimIdx_eq_prod hshape)
    intro ⟨a, b⟩ h
    rw [List.mem_iff_get] at h
    obtain ⟨n, h⟩ := h
    simp at h
    simp[←h.1, ←h.2]

def preEinsum (σ : List ℕ) (xs : List (Array × List (Fin σ.length)))
  (is : ∀ i : Fin σ.length, Fin (σ.get i)) : Option ℝ :=
  match xs with
  | [] => some 1
  | ⟨x, ns⟩ :: xs =>
    match x with
    | .float x => 
      if h₀ : x.length = (ns.map σ.get).prod then
        match preEinsum σ xs is with
        | none => none
        | some v =>
          let i₀ : List ℕ := ns.map (is ·)
          let n₀ : List ℕ := ns.map σ.get
          have h' : n₀.length = ns.length := by simp[n₀]
          if h₁ : ∀ n ∈ n₀, n ≠ 0 then
            let i : Fin x.length := (
              multiDimIdx' (List.ofFn fun (i : Fin ns.length) ↦ σ.get (ns.get i))
                (fun i ↦ (is (ns.get (i.cast <| by simp))).cast <| by simp) <| by
                simp[n₀] at h₁
                simp
                intro a
                apply h₁ ns[a]
                simp).cast <| by
                  simp [h₀]
                  congr
                  apply List.ext_get
                  · simp
                  intro n h'' h'''
                  simp
            some (v * x.get i)
          else
            none
      else none
    | _ => none

end Einsum

noncomputable def Expr.eval : Expr → List Array → Array 
  | arg i, x =>
    match x[i]? with
    | .some y => y
    | .none => .error
  | func _ f, x => f.eval x
  | idx a i, x => (a.eval x).idx (i.eval x)
  | setIdx a i b, x => (a.eval x).setIdx (i.eval x) (b.eval x)
  | add a b, x => (a.eval x).add (b.eval x)
  | sub a b, x => (a.eval x).sub (b.eval x)
  | mul a b, x => (a.eval x).mul (b.eval x)
  | div a b, x => (a.eval x).div (b.eval x)
  | divInt a b, x => (a.eval x).divInt (b.eval x)
  | mod a b, x => (a.eval x).mod (b.eval x)
  | rep n a, x => (a.eval x).rep n
  | fori_loop n a f, x =>
    match (n.eval x) with
    | .int [m] =>
      open Fin in
      let body_fun (i : ℕ) (c : Array) : Array := f.eval ((Array.int [i]) :: c :: x)
      Nat.rec (a.eval x) body_fun m.natAbs
    | _ => .error
  | eq a b, x => (a.eval x).eq (b.eval x)
  | lt a b, x => (a.eval x).lt (b.eval x)
  | select c a b, x => (c.eval x).select (a.eval x) (b.eval x)
  | toFloat a, x => (a.eval x).toFloat
  | addIdx a i b, x => (a.eval x).addIdx (i.eval x) (b.eval x)
  | sin a, x => match a.eval x with
    | .float y => .float <| y.map Real.sin
    | _ => .error
  | cos a, x => match a.eval x with
    | .float y => .float <| y.map Real.cos
    | _ => .error
  | exp a, x => match a.eval x with
    | .float y => .float <| y.map Real.exp
    | _ => .error
  | log a, x => match a.eval x with
    | .float y => .float <| y.map Real.log
    | _ => .error
  | einsum _ _, _ => .error  -- TODO: Implement einsum evaluation

end Jax
