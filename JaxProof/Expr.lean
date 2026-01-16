import JaxProof.Tensor
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.ENat.Defs

/-!
This file contains the inductive type `JAX.Expr`, which represents a jaxpr, and `JAX.Expr.code`
which generates the python code.
-/

namespace Jax

inductive Op : (ℕ∞ → Type) where
  | const_float : List ℚ → Op (some 0)
  | const_int : List ℤ → Op (some 0)
  | iota : ℕ → Op (some 0)
  | fill_int : ℕ → ℤ → Op (some 0)
  | fill_float : ℕ → ℚ → Op (some 0)
  | idx : Op (some 2)
  | setIdx : Op (some 3)
  | add : Op (some 2)
  | sub : Op (some 2)
  | mul : Op (some 2)
  | div : Op (some 2)
  | mod : Op (some 2)
  | divInt : Op (some 2)
  | rep : ℕ → Op (some 1)
  | fori_loop : Op (some 3)
  | eq : Op (some 2)
  | lt : Op (some 2)
  | select : Op (some 3)
  | toFloat : Op (some 1)
  | addIdx : Op (some 3)
  | sin : Op (some 1)
  | cos : Op (some 1)
  | exp : Op (some 1)
  | log : Op (some 1)
  | sqrt : Op (some 1)
  | einsum (s : List ℕ) : List (List (Fin s.length)) → ℕ → Op none
  | tuple : Op none
  | tupleGet : ℕ → Op (some 1)
  | anonTuple : Op none
  deriving BEq

def Op.reprType : ℕ∞ → Type
  | none => List String → String
  | some n => curryType String n

def argString {α : Type} [ToString α] (xs : List α) : String :=
  ", ".intercalate (xs.map toString)

def Op.repr {n : ℕ∞} : Op n → Op.reprType n
  | const_float x
  | const_int x => toString x
  | iota n => s!"jax.numpy.arange({n})"
  | fill_int n x => s!"jax.numpy.zeros({n}, dtype=int) + {x}"
  | fill_float n x => s!"jax.numpy.zeros({n}, dtype=float) + {x}"
  | idx => fun x i ↦ s!"{x}[{i}]"
  | setIdx => fun x i y ↦ s!"{x}.at[{i}].set({y})"
  | add => fun x y ↦ s!"{x} + {y}"
  | sub => fun x y ↦ s!"{x} - {y}"
  | mul => fun x y ↦ s!"{x} * {y}"
  | div => fun x y ↦ s!"{x} / {y}"
  | mod => fun x y ↦ s!"{x} % {y}"
  | divInt => fun x y ↦ s!"{x} // {y}"
  | rep n => fun x ↦ s!"{x}.repeat({n})"
  | fori_loop => fun n x f ↦
    s!"jax.lax.fori_loop(0, {n}[0], (lambda i, c: {f}(jax.numpy.array([i]), c)), {x})"
  | eq => fun x y ↦ s!"{x} == {y}"
  | lt => fun x y ↦ s!"{x} < {y}"
  | select => fun c x y ↦ s!"jax.lax.select({c}, {x}, {y})"
  | toFloat => fun x ↦ s!"{x}.astype(float)"
  | addIdx => fun x i y ↦ s!"{x}.at[{i}].add({y})"
  | sin => fun x ↦ s!"jax.numpy.sin({x})"
  | cos => fun x ↦ s!"jax.numpy.cos({x})"
  | exp => fun x ↦ s!"jax.numpy.exp({x})"
  | log => fun x ↦ s!"jax.numpy.log({x})"
  | sqrt => fun x ↦ s!"jax.numpy.sqrt({x})"
  | einsum s i o => fun xs ↦
    let s' := i.map (List.map s.get)
    let xs' := xs.zip (s'.zip i)
    let args := xs'.map fun ⟨x, s, i⟩ ↦ s!"{x}.reshape({argString s}), ({argString i})"
    s!"UnImplemented einsum({argString args}, ({o}))"
  | tuple => fun xs ↦ s!"tuple({argString xs})"
  | tupleGet n => fun x ↦ s!"{x}[{n}]"
  | anonTuple => fun _ ↦ ""
  
/--
The representation of JAX expressions. `Expr n` is an expression with n variables.
Each constructor's first argument is the name of the expression, which is used during code
generation
-/


inductive Expr where
  | nullop : Op (some 0) → Expr
  | unop   : Op (some 1) → Expr → Expr
  | binop  : Op (some 2) → Expr → Expr → Expr
  | triop  : Op (some 3) → Expr → Expr → Expr → Expr
  | varop  : Op none → List Expr → Expr
  | arg : ℕ → Expr
  | fn : ℕ → Expr → Expr
  deriving BEq

def Expr.toString : Expr → String
  | nullop op => op.repr
  | unop op x => op.repr x.toString
  | binop op x y => op.repr x.toString y.toString
  | triop op x y z => op.repr x.toString y.toString z.toString
  | varop op xs => op.repr (xs.map Expr.toString)
  | arg n => s!"(arg {n})"
  | fn n x => s!"(fn {n} {x.toString})"

instance : ToString Expr where toString := Expr.toString

def Expr.lift (n : ℕ) (e : Expr) : Expr :=
  go 0 e
where go (m : ℕ) : Expr → Expr
  | arg l => if l < m then arg l else arg (l + n)
  | fn l f => fn l (go (m + l) f)
  | nullop op => nullop op
  | unop op x => unop op (go m x)
  | binop op x y => binop op (go m x) (go m y)
  | triop op x y z => triop op (go m x) (go m y) (go m z)
  | varop op xs => varop op (xs.map (go m))

def Expr.lower (n : ℕ) (e : Expr) : Option Expr :=
  go 0 e
where go (m : ℕ) : Expr → Option Expr
  | arg l => if l < m then some (arg l) else if (m + n) ≤ l then some (arg (l - n)) else none
  | fn l f => match go (m + l) f with
    | some f => fn l f
    | none => none
  | nullop op => nullop op
  | unop op x => match go m x with
    | some x => unop op x
    | none => none
  | binop op x y => match go m x, go m y with
    | some x, some y => binop op x y
    | _, _ => none
  | triop op x y z => match go m x, go m y, go m z with
    | some x, some y, some z => triop op x y z
    | _, _, _ => none
  | varop op xs =>
    let rec go' (done : List Expr) : List (Option Expr) → Option (List Expr)
    | [] => done
    | some x :: xs => go' (done.concat x) xs
    | _ => none
    match go' [] (xs.map (go m)) with
    | some xs => varop op xs
    | none => none

def Expr.lowerable (n : ℕ) (e : Expr) : List Expr :=
  match e.lower n with
  | some e => [e]
  | none => match e with
    | unop _ x => x.lowerable n
    | binop _ x y => x.lowerable n ++ y.lowerable n
    | triop _ x y z => x.lowerable n ++ y.lowerable n ++ z.lowerable n
    | varop _ xs => xs.foldr (fun x l ↦ x.lowerable n ++ l) []
    | _ => []

def Expr.outward : Expr → Expr
  | unop op x => unop op x.outward
  | binop op x y => binop op x.outward y.outward
  | triop op x y z => triop op x.outward y.outward z.outward
  | varop op xs => varop op (xs.map Expr.outward)
  | fn n f =>
    let f' := f.outward
    match f'.lowerable n with
    | [] => fn n f'
    | l => unop (.tupleGet 0) (varop .tuple ((fn n f') :: l))
  | e => e

structure CodeGenCtx where
  vars : List Expr
  code : List String

def addExpr (expr : Expr) : StateM CodeGenCtx String := do
  let ⟨vars, code⟩ ← get
  let name := s!"x{vars.length}"
  set (CodeGenCtx.mk (vars.concat expr) code)
  return name

def writeLines (lines : List String) : StateM CodeGenCtx Unit := do
  let ⟨vars, code⟩ ← get
  set (CodeGenCtx.mk vars (code ++ lines))

/-
TODO: In a subfunction, some expressions that does not depend on the newly introduced args
can be evaluated outside.
-/
def Expr.genCode (expr : Expr) : StateM CodeGenCtx String := do
  let ⟨vars, _⟩ ← get
  match vars.idxOf? expr with
  | .some i => return s!"x{i}"
  | .none =>
    match expr with
    | arg _ =>  addExpr expr
    | unop (.tupleGet 0) (varop .tuple (x :: xs)) =>
      let _ ← xs.mapM Expr.genCode
      x.genCode
    | fn n x =>
      let fname ← addExpr expr
      let ⟨vars, _⟩ ← get
      let ctx : CodeGenCtx := ⟨vars.map (Expr.lift n), []⟩
      let (name, ⟨sub_vars, sub_code⟩) := x.genCode ctx
      let body : List String := (sub_code.concat s!"return {name}").map ("  " ++ ·)
      let argnames : List String := (List.range n).map fun i ↦
        match sub_vars.idxOf? (Expr.arg i) with
        | .some j => s!"x{j}"
        | .none => s!"_x{i}"
      let head : String := s!"def {fname}(" ++ ", ".intercalate argnames ++ "):"
      writeLines (head :: body)
      return fname
    | nullop op =>
      let name ← addExpr expr
      writeLines [s!"{name} = {@id String op.repr}"]
      return name
    | unop op x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {@id String (op.repr xname)}"]
      return name
    | binop op x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {@id String (op.repr xname yname)}"]
      return name
    | triop op x y z =>
      let xname ← x.genCode
      let yname ← y.genCode
      let zname ← z.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {@id String (op.repr xname yname zname)}"]
      return name
    | varop op xs =>
      let xnames ← xs.mapM Expr.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {@id String (op.repr xnames)}"]
      return name

def Expr.code (expr : Expr) : String := "\n".intercalate (expr.genCode ⟨[], []⟩).2.2

end Jax
