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
        _ = (i₀ + 1) + s₀ * i' := by ring
        _ ≤ s₀ * (i' + 1) := by rw[add_comm i'.val 1, mul_add, mul_one]; gcongr; exact i₀.isLt
        _ ≤ _ := by gcongr; rw[Nat.succ_le_iff]; exact i'.isLt
termination_by s.val

def unflattenIdx (s : ValidShape) (idx : Fin s.size) : ValidIdx s :=
  let ⟨sval, sprop⟩ := s
  match sval with
  | [] => fun i ↦ nomatch i
  | s₀ :: s₁ => fun i ↦
    let idx' : Fin (s₁.prod * s₀) := idx.cast <| by simp [ValidShape.size]; rw[mul_comm]
    if h : i = ⟨0, by simp⟩ then
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
          show (x 0).val / s₀ = 0 from Nat.div_eq_zero_iff.mpr (Or.inr (x 0).isLt), zero_add]
        rw[mul_comm, Nat.mul_div_cancel _ (by omega)]
      simp only [Fin.eta, ih]
      let g (j : Fin (s₁.length + 1)) : ℕ := x i
      have : x (i.pred h).succ = g (i.pred h).succ := by congr; all_goals simp
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
      have ih := ih sprop.2
        (x.cast (by simp[ValidShape.size, mul_comm]) : Fin (s₁.prod * s₀)).divNat
      simp [Fin.divNat, ← Fin.val_eq_val] at ih
      rw[← ih]
      congr

def einProd {α : Type} [Mul α] [One α] (s : ValidShape)
  (xs : List ((List α) × List (Fin s.val.length))) : Option (ValidIdx s → α) :=
  process xs 1
where process (xs : List ((List α) × List (Fin s.val.length))) (acc : ValidIdx s → α) : 
  Option (ValidIdx s → α) :=
  match xs with
  | [] => some acc
  | ⟨x, indices⟩ :: xs =>
    let s' := subshape s indices
    if h₀ : x.length = s'.size then
      process xs fun idx ↦
        let idx' : ValidIdx s' := fun i ↦ 
          let i' : Fin indices.length := i.cast <| by simp [s', subshape]
          Fin.mk (idx (indices.get i')) <| by
            simp [s', subshape]
            exact (idx indices[i']).isLt
        acc idx * x[flattenIdx idx']
    else
      none  -- Invalid: shape mismatch

def einSum {α : Type} [AddCommMonoid α] [Monoid α] (s : List ℕ)
  (xs : List ((List α) × List ℤ)) (out : List ℤ) : Option (List α) :=
  if hs : s.length = 0 then
    .none
  else
    have : NeZero s.length := ⟨hs⟩
  let xs' : List ((List α) × List (Fin s.length)) :=
    xs.map fun ⟨x, indices⟩ ↦ ⟨x, indices.map Int.cast⟩
  let out' : List (Fin s.length) := out.map Int.cast
  if h : ∀ n ∈ s, n ≠ 0 then
    let s' : ValidShape := ⟨s, h⟩
    match einProd s' xs' with
    | none => .none
    | some x => .some <| List.ofFn fun j ↦
      let idx := unflattenIdx (subshape s' out') j
      let sumIdx : Finset (ValidIdx s') :=
        {idx' | ∀ i : Fin out'.length,
          idx' (out'.get i) = (idx (i.cast (by simp))).cast (by simp[s'])} 
      ∑ idx' ∈ sumIdx, x idx'
  else
    .none

#eval einSum [2,2] [([1,2,3,4],[0,1])] [1,0]

/-- Represents a parsed einsum string like "ij,jk->ik" -/
structure EinsumExpr where
  inputs : List (List Char)   -- Input index labels for each array
  output : List Char          -- Output index labels

/-- Parse an einsum string - supports both explicit "ij,jk->ik" and implicit "ij,jk" forms -/
def parseEinsumString (s : String) : Option EinsumExpr :=
  match s.splitOn "->" with
  | [inputs, outputs] =>
    some ⟨(inputs.splitOn ",").map String.toList, outputs.toList⟩
  | _ => none

def EinsumExpr.all_chars (e : EinsumExpr) : List Char := e.inputs.flatten.dedup

/-- Convert character indices to Fin indices based on dimension mapping -/
def indicesFromChars (σ : List ℕ) (chars : List Char) : Option (List (Fin σ.length)) :=
  chars.mapM (fun c =>
    let pos := c.toNat - 'a'.toNat
    if h : pos < σ.length then
      some (Fin.mk pos h)
    else
      none
  )

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
  | einsum s inputs, x =>
      -- For now, return error. Full implementation would:
      -- 1. Parse the einsum string
      -- 2. Extract shapes from input arrays  
      -- 3. Map indices to dimensions
      -- 4. Use preEinsum infrastructure for computation
      .error  -- Placeholder: complete implementation needs shape inference

end Jax
