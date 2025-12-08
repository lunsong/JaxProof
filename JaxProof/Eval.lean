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
  | tuple : List Array → Array
  | error : Array

def toFin (n : ℕ) (xs : List ℤ) : Option (List (Fin n)) :=
  if h : ∀ x ∈ xs, 0 ≤ x ∧ x < n then
    some <| .ofFn fun i ↦ Fin.mk (xs.get i).toNat <| by
      have := h _ (List.get_mem xs i)
      exact (Int.toNat_lt this.1).mpr this.2
  else
    none

def Array.idx : Array → Array → Array
  | float x, int i =>
    match toFin x.length i with
    | none => error
    | some i => float (i.map x.get)
  | int x, int i =>
    match toFin x.length i with
    | none => error
    | some i => int (i.map x.get)
  | _, _ => error

def Array.setIdx : Array → Array → Array → Array
  | float x, int i, float y =>
    match toFin x.length i with
    | none => error
    | some i =>
      if ¬ i.Nodup then error else
      if hiy : i.length = y.length then
        float <| .ofFn fun a ↦ match i.finIdxOf? a with
          | none => x.get a
          | some j => y.get (j.cast hiy)
      else
        error
  | int x, int i, int y =>
    match toFin x.length i with
    | none => error
    | some i =>
      if ¬ i.Nodup then error else
      if hiy : i.length = y.length then
        int <| .ofFn fun a ↦ match i.finIdxOf? a with
          | none => x.get a
          | some j => y.get (j.cast hiy)
      else
        error
  | _, _, _ => error

def Array.pairwise (intFn : ℤ → ℤ → ℤ) (realFn : ℝ → ℝ → ℝ) : Array → Array → Array
  | int x, int y =>
    if h : x.length = y.length then
      int <| .ofFn fun i ↦ intFn (x.get i) (y.get (i.cast h))
    else
      error
  | float x, float y =>
    if h : x.length = y.length then
      float <| .ofFn fun i ↦ realFn (x.get i) (y.get (i.cast h))
    else
      error
  | _, _ => error


def Array.add := Array.pairwise (· + ·) (· + ·)
def Array.sub := Array.pairwise (· - ·) (· - ·)
def Array.mul := Array.pairwise (· * ·) (· * ·)

noncomputable def Array.div : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      if 0 ∈ y then
        error
      else
        float <| .ofFn <| fun (i : Fin x.length) => x.get i / y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.divInt : Array → Array → Array
  | int x, int y =>
    if h : x.length = y.length then
      if 0 ∈ y then
        error
      else
        int <| .ofFn <| fun (i : Fin x.length) => x.get i / y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.mod : Array → Array → Array
  | int x, int y =>
    if h : x.length = y.length then
      if 0 ∈ y then
        error
      else
        int <| .ofFn <| fun (i : Fin x.length) => x.get i % y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.toFloat : Array → Array
  | int x => float (x.map Int.cast)
  | _ => error

def Array.rep (n : ℕ) : Array → Array
  | float x => float <| .ofFn <| fun (i : Fin (x.length * n)) => x.get i.divNat
  | int x => int <| .ofFn <| fun (i : Fin (x.length * n)) => x.get i.divNat
  | _ => error

def Array.eq : Array → Array → Array
  | int x, int y =>
    if h : x.length = y.length then
      int <| .ofFn <| fun (i : Fin x.length) => if x.get i = y.get (i.cast h) then 1 else 0
    else
      error
  | _, _ => error
 
noncomputable def Array.lt : Array → Array → Array
  | float x, float y =>
    if h : x.length = y.length then
      int <| .ofFn <| fun (i : Fin x.length) => if x.get i < y.get (i.cast h) then 1 else 0
    else
      error
  | int x, int y =>
    if h : x.length = y.length then
      int <| .ofFn <| fun (i : Fin x.length) => if x.get i < y.get (i.cast h) then 1 else 0
    else
      error
  | _, _ => error

def Array.select : Array → Array → Array → Array
  | int c, float x, float y =>
    if h : c.length = x.length ∧ c.length = y.length then
      float <| .ofFn <| fun (i : Fin c.length) =>
        if c.get i = 0 then y.get (i.cast h.2) else x.get (i.cast h.1)
    else
      error
  | int c, int x, int y =>
    if h : c.length = x.length ∧ c.length = y.length then
      int <| .ofFn <| fun (i : Fin c.length) =>
        if c.get i = 0 then y.get (i.cast h.2) else x.get (i.cast h.1)
    else
      error
  | _, _, _ => error

def addIdx {α : Type} [AddCommMonoid α] (x y : List α) (i : List ℤ) : Option (List α) :=
  match toFin x.length i with
  | none => none
  | some i =>
    if h : i.length = y.length then
      some <| .ofFn fun j ↦ ∑ k with i.get k = j, y.get (k.cast h)
    else
      none

nonrec def Array.addIdx : Array → Array → Array → Array
  | float x, int i, float y =>
    match addIdx x y i with
    | some z => float z
    | none => error
  | int x, int i, int y =>
    match addIdx x y i with
    | some z => int z
    | none => error
  | _, _, _ => error

namespace Einsum

variable {α : Type} [AddCommMonoid α] [Monoid α] 

def prod (s : ValidShape) (xs : List ((List α) × List (Fin s.val.length))) :
    Option (ValidIdx s → α) :=
  go xs 1
where go (xs : List ((List α) × List (Fin s.val.length))) (acc : ValidIdx s → α) : 
  Option (ValidIdx s → α) :=
  match xs with
  | [] => some acc
  | ⟨x, indices⟩ :: xs =>
    let s' := subshape s indices
    if h₀ : x.length = s'.size then
      go xs fun idx ↦
        let idx' : ValidIdx s' := fun i ↦ 
          let i' : Fin indices.length := i.cast <| by simp [s', subshape]
          Fin.mk (idx (indices.get i')) <| by
            simp [s', subshape]
            exact (idx indices[i']).isLt
        acc idx * x[flattenIdx idx']
    else
      none  -- Invalid: shape mismatch

def sum (s : ValidShape) (xs : List ((List α) × List (Fin s.val.length)))
    (out : List (Fin s.val.length)) : Option (List α) :=
  match prod s xs with
  | none => .none
  | some x => .some <| .ofFn fun j ↦
    let idx := unflattenIdx (subshape s out) j
    let sumIdx : Finset (ValidIdx s) :=
      {idx' | ∀ i , idx' (out.get i) = (idx (i.cast (by simp))).cast (by simp)} 
    ∑ idx' ∈ sumIdx, x idx'

end Einsum

def allFloat (xs : List Array) : Option (List (List ℝ)) :=
  go xs []
where go : List Array → List (List ℝ) → Option (List (List ℝ))
  | [], ys => some ys
  | .float x :: xs, ys => go xs (ys.concat x)
  | _, _ => none

def Array.einsum (s : List ℕ) (i : List (List (Fin s.length))) (o : List (Fin s.length))
    (xs : List Array) : Array :=
  if h : ∀ n ∈ s, n ≠ 0 then
    match allFloat xs with
    | none => .error
    | some xs => match Einsum.sum ⟨s, h⟩ (xs.zip i) o with
      | none => .error
      | some ys => .float ys
  else
    .error

/-
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
      let body_fun (i : ℕ) (c : Array) : Array := f.eval (Array.int [i] :: c :: x)
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
  | einsum s i o as, x => Array.einsum s i o (as.map (Expr.eval · x))
  | tuple as, x =>  .tuple (as.map (Expr.eval · x))
  | tupleGet i a, x => match a.eval x with
    | .tuple a =>
      if h : i < a.length then
        a.get ⟨i, h⟩
      else .error
    | _ => .error
-/

end Jax
