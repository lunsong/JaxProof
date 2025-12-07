import JaxProof.MultiDimIdx
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Tuple.Basic

/-!
This file contains the inductive type `JAX.Expr`, which represents a jaxpr, and `JAX.Expr.code`
which generates the python code.
-/

namespace Jax

/--
The representation of JAX expressions. `Expr n` is an expression with n variables.
Each constructor's first argument is the name of the expression, which is used during code
generation
-/
inductive Expr where
  /-- `arg i` is the i'th argument -/
  | arg : ℕ → Expr
  | func : ℕ → Expr → Expr
  /-- indexing an array, a = b[i] -/
  | idx : Expr → Expr → Expr
  /-- set items of an array, a = b.set[i].set(c) -/
  | setIdx : Expr → Expr → Expr → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | mod : Expr → Expr → Expr
  | divInt : Expr → Expr → Expr
  /-- repeat an array, `jax.numpy.repeat` -/
  | rep : ℕ → Expr → Expr
  /-- jax.lax.fori_loop -/
  | fori_loop : Expr → Expr → Expr → Expr
  | eq : Expr → Expr → Expr
  | lt : Expr → Expr → Expr
  | select : Expr → Expr → Expr → Expr
  | toFloat : Expr → Expr
  | addIdx : Expr → Expr → Expr → Expr
  | sin : Expr → Expr
  | cos : Expr → Expr
  | exp : Expr → Expr
  | log : Expr → Expr
  | einsum (s : List ℕ) : List (List (Fin s.length)) → List (Fin s.length) → List Expr → Expr
  | tuple : List Expr → Expr
  | tupleGet : ℕ → Expr → Expr
deriving BEq

def Expr.lift : Expr → Expr
  | arg i => arg i.succ
  | func n x => func n x.lift
  | idx x i => idx x.lift i.lift
  | setIdx x i y => setIdx x.lift i.lift y.lift
  | add x y => add x.lift y.lift
  | sub x y => sub x.lift y.lift
  | mul x y => mul x.lift y.lift
  | div x y => div x.lift y.lift
  | divInt x y => divInt x.lift y.lift
  | mod x y => mod x.lift y.lift
  | rep n x => rep n x.lift
  | fori_loop n x f => fori_loop n.lift x.lift f.lift
  | eq x y => eq x.lift y.lift
  | lt x y => lt x.lift y.lift
  | select c x y => select c.lift x.lift y.lift
  | toFloat x => toFloat x.lift
  | addIdx x i y => addIdx x.lift i.lift y.lift
  | sin x => sin x.lift
  | cos x => cos x.lift
  | exp x => exp x.lift
  | log x => log x.lift
  | einsum s i o xs => einsum s i o (xs.map Expr.lift)
  | tuple xs => tuple (xs.map Expr.lift)
  | tupleGet n x => tupleGet n x.lift

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

def argString : List String → String := ", ".intercalate

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
    | .arg _ =>  addExpr expr
    | .func n x =>
      let fname ← addExpr x
      let ⟨vars, _⟩ ← get
      let ctx : CodeGenCtx := ⟨vars.map (Nat.repeat Expr.lift n), []⟩
      let (name, ⟨sub_vars, sub_code⟩) := x.genCode ctx
      let body : List String := (sub_code.concat s!"return {name}").map ("  " ++ ·)
      let argnames : List String := (List.range n).map fun i ↦
        match sub_vars.idxOf? (Expr.arg i) with
        | .some j => s!"x{j}"
        | .none => s!"_x{i}"
      let head : String := s!"def {fname}(" ++ ", ".intercalate argnames ++ "):"
      writeLines (head :: body)
      return fname
    | .add x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} + {yname}"]
      return name
    | .sub x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} - {yname}"]
      return name
    | .div x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} / {yname}"]
      return name
    | .divInt x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} // {yname}"]
      return name
    | .mul x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} * {yname}"]
      return name
    | .mod x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} % {yname}"]
      return name
    | .idx x i =>
      let xname ← x.genCode
      let iname ← i.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname}[{iname}]"]
      return name
    | .setIdx x i y =>
      let xname ← x.genCode
      let iname ← i.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname}.at[{iname}].set({yname})"]
      return name
    | .rep n x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname}.repeat({n})"]
      return name
    | .fori_loop n x f =>
      let xname ← x.genCode
      let nname ← n.genCode
      let fname ← f.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = jax.lax.fori_loop(0, {nname}[0], {fname}, {xname})"]
      return name
    | .eq x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} == {yname}"]
      return name
    | .lt x y =>
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname} < {yname}"]
      return name
    | .select c x y =>
      let cname ← c.genCode
      let xname ← x.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = jax.lax.select({cname}, {xname}, {yname})"]
      return name
    | .toFloat x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname}.astype(float)"]
      return name
    | .addIdx x i y =>
      let xname ← x.genCode
      let iname ← i.genCode
      let yname ← y.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname}.at[{iname}].add({yname})"]
      return name
    | .sin x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = jax.numpy.sin({xname})"]
      return name
    | .cos x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = jax.numpy.cos({xname})"]
      return name
    | .exp x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = jax.numpy.exp({xname})"]
      return name
    | .log x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = jax.numpy.log({xname})"]
      return name
    | .einsum shape indices out xs =>
      let xnames ← xs.mapM Expr.genCode
      let indices_and_shapes := indices.map fun idx ↦ (idx.map Fin.val, idx.map shape.get)
      let args := (xnames.zip indices_and_shapes).map fun ⟨name, indices, shape⟩ ↦
        s!"{name}.reshape({argString (shape.map toString)}), ({argString (indices.map toString)})"
      let name ← addExpr expr
      writeLines [s!"{name} = jax.numpy.einsum({argString args}, ({argString (out.map toString)}))"]
      return name
    | .tuple xs =>
      let xnames ← xs.mapM Expr.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = tuple({argString xnames})"]
      return name
    | .tupleGet n x =>
      let xname ← x.genCode
      let name ← addExpr expr
      writeLines [s!"{name} = {xname}[{n}]"]
      return name


def Expr.code (expr : Expr) : String := "\n".intercalate (expr.genCode ⟨[], []⟩).2.2

end Jax
