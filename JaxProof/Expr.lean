import JaxProof.MultiDimIdx
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
  | einsum (s : List ℕ) : List (List (Fin s.length)) → List (Fin s.length) → Op none
  | tuple : Op none
  | tupleGet : ℕ → Op (some 1)
  | anonTuple : Op none
  deriving DecidableEq

def Op.reprType : ℕ∞ → Type
  | none => List String → String
  | some n => curryType String n

def argString {α : Type} [ToString α] (xs : List α) : String :=
  ", ".intercalate (xs.map toString)

def Op.repr {n : ℕ∞} : Op n → Op.reprType n
  | const_float x
  | const_int x => toString x
  | idx => fun x i ↦ s!"{x}[{i}]"
  | setIdx => fun x i y ↦ s!"{x}.at[{i}].set({y})"
  | add => fun x y ↦ s!"{x} + {y}"
  | sub => fun x y ↦ s!"{x} - {y}"
  | mul => fun x y ↦ s!"{x} * {y}"
  | div => fun x y ↦ s!"{x} / {y}"
  | mod => fun x y ↦ s!"{x} % {y}"
  | divInt => fun x y ↦ s!"{x} // {y}"
  | rep n => fun x ↦ s!"{x}.repeat({n})"
  | fori_loop => fun n x f ↦ s!"jax.lax.fori_loop(0, {n}[0], {f}, {x})"
  | eq => fun x y ↦ s!"{x} == {y}"
  | lt => fun x y ↦ s!"{x} < {y}"
  | select => fun c x y ↦ s!"jax.lax.select({c}, {x}, {y})"
  | toFloat => fun x ↦ s!"{x}.astype(float)"
  | addIdx => fun x i y ↦ s!"{x}.at[{i}].add({y})"
  | sin => fun x ↦ s!"jax.numpy.sin({x})"
  | cos => fun x ↦ s!"jax.numpy.cos({x})"
  | exp => fun x ↦ s!"jax.numpy.exp({x})"
  | log => fun x ↦ s!"jax.numpy.log({x})"
  | einsum s i o => fun xs ↦
    let s' := i.map (List.map s.get)
    let xs' := xs.zip (s'.zip i)
    let args := xs'.map fun ⟨x, s, i⟩ ↦ s!"{x}.reshape({argString s}), ({argString i})"
    s!"jax.numpy.einsum({argString args}, ({argString o}))"
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
  | quadop : Op (some 4) → Expr → Expr → Expr → Expr → Expr
  | varop  : Op none → List Expr → Expr
  | arg : ℕ → Expr
  | fn : ℕ → Expr → Expr
  deriving BEq

-- The manual instance definition
def Expr.decEq (a b : Expr) : Decidable (a = b) :=
  match a, b with
  -- 1. Handle matching constructors
  | nullop o1, nullop o2 =>
    if h : o1 = o2 then isTrue (congrArg nullop h)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  | unop o1 e1, unop o2 e2 =>
    if h_op : o1 = o2 then
      match Expr.decEq e1 e2 with
      | isTrue h_e => isTrue (by rw [h_op, h_e])
      | isFalse h_e => isFalse (fun h_eq => by injection h_eq; contradiction)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  | binop o1 e1 f1, binop o2 e2 f2 =>
    if h_op : o1 = o2 then
      match Expr.decEq e1 e2 with
      | isTrue h_e =>
        match Expr.decEq f1 f2 with
        | isTrue h_f => isTrue (by rw [h_op, h_e, h_f])
        | isFalse h_f => isFalse (fun h_eq => by injection h_eq; contradiction)
      | isFalse h_e => isFalse (fun h_eq => by injection h_eq; contradiction)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  | triop o1 e1 f1 g1, triop o2 e2 f2 g2 =>
    if h_op : o1 = o2 then
      match Expr.decEq e1 e2 with
      | isTrue h_e =>
        match Expr.decEq f1 f2 with
        | isTrue h_f =>
          match Expr.decEq g1 g2 with
          | isTrue h_g => isTrue (by rw [h_op, h_e, h_f, h_g])
          | isFalse h_g => isFalse (fun h_eq => by injection h_eq; contradiction)
        | isFalse h_f => isFalse (fun h_eq => by injection h_eq; contradiction)
      | isFalse h_e => isFalse (fun h_eq => by injection h_eq; contradiction)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  | quadop o1 e1 f1 g1 h1, quadop o2 e2 f2 g2 h2 =>
    if h_op : o1 = o2 then
      match Expr.decEq e1 e2 with
      | isTrue h_e =>
        match Expr.decEq f1 f2 with
        | isTrue h_f =>
          match Expr.decEq g1 g2 with
          | isTrue h_g =>
            match Expr.decEq h1 h2 with
            | isTrue h_h => isTrue (by rw [h_op, h_e, h_f, h_g, h_h])
            | isFalse h_h => isFalse (fun h_eq => by injection h_eq; contradiction)
          | isFalse h_g => isFalse (fun h_eq => by injection h_eq; contradiction)
        | isFalse h_f => isFalse (fun h_eq => by injection h_eq; contradiction)
      | isFalse h_e => isFalse (fun h_eq => by injection h_eq; contradiction)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  -- This is the tricky case: we use the helper `decEqList` defined below
  | varop o1 l1, varop o2 l2 =>
    if h_op : o1 = o2 then
      match decEqList l1 l2 with
      | isTrue h_l => isTrue (by rw [h_op, h_l])
      | isFalse h_l => isFalse (fun h_eq => by injection h_eq; contradiction)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  | arg n1, arg n2 =>
    if h : n1 = n2 then isTrue (congrArg arg h)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  | fn n1 e1, fn n2 e2 =>
    if h_n : n1 = n2 then
      match Expr.decEq e1 e2 with
      | isTrue h_e => isTrue (by rw [h_n, h_e])
      | isFalse h_e => isFalse (fun h_eq => by injection h_eq; contradiction)
    else isFalse (fun h_eq => by injection h_eq; contradiction)

  -- 2. Handle mismatched constructors
  -- (We can group these using wildcards for brevity)
  | nullop _, unop _ _ 
  | nullop _, binop _ _ _ 
  | nullop _, triop _ _ _ _ 
  | nullop _, quadop _ _ _ _ _ 
  | nullop _, varop _ _ 
  | nullop _, arg _ 
  | nullop _, fn _ _ 

  | unop _ _, nullop _ 
  | unop _ _, binop _ _ _ 
  | unop _ _, triop _ _ _ _ 
  | unop _ _, quadop _ _ _ _ _ 
  | unop _ _, varop _ _ 
  | unop _ _, arg _ 
  | unop _ _, fn _ _ 

  | binop _ _ _, nullop _ 
  | binop _ _ _, unop _ _ 
  | binop _ _ _, triop _ _ _ _ 
  | binop _ _ _, quadop _ _ _ _ _ 
  | binop _ _ _, varop _ _ 
  | binop _ _ _, arg _ 
  | binop _ _ _, fn _ _ 

  | triop _ _ _ _, nullop _ 
  | triop _ _ _ _, unop _ _ 
  | triop _ _ _ _, binop _ _ _ 
  | triop _ _ _ _, quadop _ _ _ _ _ 
  | triop _ _ _ _, varop _ _ 
  | triop _ _ _ _, arg _ 
  | triop _ _ _ _, fn _ _ 

  | quadop _ _ _ _ _, nullop _ 
  | quadop _ _ _ _ _, unop _ _ 
  | quadop _ _ _ _ _, binop _ _ _ 
  | quadop _ _ _ _ _, triop _ _ _ _ 
  | quadop _ _ _ _ _, varop _ _ 
  | quadop _ _ _ _ _, arg _ 
  | quadop _ _ _ _ _, fn _ _ 

  | varop _ _, nullop _ 
  | varop _ _, unop _ _ 
  | varop _ _, binop _ _ _ 
  | varop _ _, triop _ _ _ _ 
  | varop _ _, quadop _ _ _ _ _ 
  | varop _ _, arg _ 
  | varop _ _, fn _ _ 

  | arg _, nullop _ 
  | arg _, unop _ _ 
  | arg _, binop _ _ _ 
  | arg _, triop _ _ _ _ 
  | arg _, quadop _ _ _ _ _ 
  | arg _, varop _ _ 
  | arg _, fn _ _ 

  | fn _ _, nullop _ 
  | fn _ _, unop _ _ 
  | fn _ _, binop _ _ _ 
  | fn _ _, triop _ _ _ _ 
  | fn _ _, quadop _ _ _ _ _ 
  | fn _ _, varop _ _ 
  | fn _ _, arg _ => isFalse Expr.noConfusion

where
  -- Helper: Compares two lists of Exprs by recursively calling Expr.decEq
  decEqList (l1 l2 : List Expr) : Decidable (l1 = l2) :=
    match l1, l2 with
    | [], [] => isTrue rfl
    | [], _::_ => isFalse List.noConfusion
    | _::_, [] => isFalse List.noConfusion
    | h1::t1, h2::t2 =>
      match Expr.decEq h1 h2 with
      | isFalse h => isFalse (fun h_eq => by injection h_eq; contradiction)
      | isTrue h_head =>
        match decEqList t1 t2 with
        | isFalse h => isFalse (fun h_eq => by injection h_eq; contradiction)
        | isTrue h_tail => isTrue (List.cons_eq_cons.mpr ⟨h_head, h_tail⟩)

-- Register the instance
instance : DecidableEq Expr := Expr.decEq

def Expr.lift (n : ℕ) (e : Expr) : Expr :=
  go 0 e
where go (m : ℕ) : Expr → Expr
  | arg l => if l < m then arg l else arg (l + n)
  | fn l f => fn m (go (m + l) f)
  | nullop op => nullop op
  | unop op x => unop op (go m x)
  | binop op x y => binop op (go m x) (go m y)
  | triop op x y z => triop op (go m x) (go m y) (go m z)
  | quadop op x y z w => quadop op (go m x) (go m y) (go m z) (go m w)
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
  | quadop op x y z w => match go m x, go m y, go m z, go m w with
    | some x, some y, some z, some w => quadop op x y z w
    | _, _, _, _ => none
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
    | binop _ x y => (x.lowerable n ++ y.lowerable n).dedup
    | triop _ x y z => (x.lowerable n ++ y.lowerable n ++ z.lowerable n).dedup
    | quadop _ x y z w => (x.lowerable n ++ y.lowerable n ++ z.lowerable n ++ w.lowerable n).dedup
    | _ => []

def Expr.outward : Expr → Expr
  | unop op x => unop op x.outward
  | binop op x y => binop op x.outward y.outward
  | triop op x y z => triop op x.outward y.outward z.outward
  | quadop op x y z w => quadop op x.outward y.outward z.outward w.outward
  | varop op xs => varop op (xs.map Expr.outward)
  | fn n f =>
    let f' := f.outward
    match f'.lowerable n with
    | [] => fn n f'
    | l => unop (.tupleGet 0) (varop .tuple ((fn n f') :: l))
  | e => e

  /-
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
-/

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
