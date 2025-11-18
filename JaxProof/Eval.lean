import JaxProof.Expr

namespace JAX

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
    let i' : List (Fin x.length) := i
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
    let i' : List (Fin x.length) := i
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
  | int x, int y =>
    if h : x.length = y.length then
      if 0 ∈ y then
        error
      else
        int <| List.ofFn <| fun (i : Fin x.length) => x.get i / y.get (i.cast h)
    else
      error
  | _, _ => error

def Array.rep (n : ℕ) : Array → Array
  | float x => float <| List.ofFn <| fun (i : Fin (x.length * n)) => x.get i.divNat
  | int x => int <| List.ofFn <| fun (i : Fin (x.length * n)) => x.get i.divNat
  | error => error
      
noncomputable def Expr.eval {n : ℕ} : Expr n →  (Fin n → Array) → Array 
  | arg _ i, x => x i
  | const_int _ a, _ => Array.int a
  | const_float _ a, _ => Array.float a
  | idx _ a i, x => (a.eval x).idx (i.eval x)
  | setIdx _ a i b, x => (a.eval x).setIdx (i.eval x) (b.eval x)
  | add _ a b, x => (a.eval x).add (b.eval x)
  | sub _ a b, x => (a.eval x).sub (b.eval x)
  | mul _ a b, x => (a.eval x).mul (b.eval x)
  | div _ a b, x => (a.eval x).div (b.eval x)
  | rep _ n a, x => (a.eval x).rep n
  | fori_loop _ n a f, x =>
    open Fin in
    let body_fun (i : ℕ) (c : Array) : Array := f.eval (cons (Array.int [i]) (cons c x))
    Nat.rec (a.eval x) body_fun n

def curryType : ℕ → Type
  | 0 => Array
  | n + 1 => Array → curryType n

def curry {n : ℕ} (f : (Fin n → Array) → Array) : curryType n :=
  match n with
  | 0 => f (fun i ↦ nomatch i)
  | _ + 1 => fun x ↦ curry <| f ∘ (Fin.cons x ·)

noncomputable def Expr.eval' {n : ℕ} (expr : Expr n) : curryType n := curry expr.eval

end JAX
