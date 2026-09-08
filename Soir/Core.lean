import Soir.Curry
import Soir.Meta

/-!
# The Core of the Soir framework

This file contains definitions for code generation and native evaluation
-/

namespace Soir

/-- `OpType` specifies the primitive ops. First-order and second-order ops are put together. -/
def OpType (data : Type) : Type 1 := List (List data × List data) → List data → List data → Type

/-- `Expr` represent an Soir expression with multiput input and multiple output, using `data`
as the data type and `op` as primitive ops. -/
inductive Expr {data : Type} (op : OpType data) :
    List data → List data → Type where
  | nil {args : List data} : Expr op args []
  | append {args outs outs' : List data} :
    Expr op args outs → Expr op args outs' → Expr op args (outs ++ outs')
  | select {args outs : List data} (i : List (Fin outs.length)) :
    Expr op args outs → Expr op args (i.map outs.get)
  | arg {args : List data} (i : Fin args.length) : Expr op args [args[i]]
  | apply {args ins outs : List data} : Expr op ins outs → Expr op args ins → Expr op args outs
  | bind {args ins outs : List data} {exprs : List (List data × List data)} :
    op exprs ins outs →
      (∀ i : Fin exprs.length, Expr op exprs[i].1 exprs[i].2) → Expr op args ins → Expr op args outs

section

variable {data : Type} {op : OpType data} [∀ exprs, ∀ ins, ∀ outs, ToString (op exprs ins outs)]

abbrev Cached (α : Type) : Type := List (UInt64 × α)

abbrev Expr.CodeM (α : Type) : Type :=
  StateM (Nat × Cached (List Nat × String) × Cached String) α

/-- Structural hash of an `Expr`, used as the cache key for sub-expression
deduplication during code generation. Two structurally equal expressions always
hash to the same value, so deduplication no longer relies on pointer identity
and the generated code is reproducible across builds.

This is a hash, not a full equality test: it optimistically merges expressions
with the same hash. Collisions are astronomically unlikely for realistic
programs (64-bit hash space), and the worst case is incorrect codegen on a
collision — acceptable given the "catch most cases" goal. -/
def Expr.hashExpr {args outs : List data} : Expr op args outs → UInt64
  | .nil => hash (0 : Nat)
  | .arg i => hash (1, i.val)
  | .append xs ys => hash (2, (xs.hashExpr, ys.hashExpr))
  | .select is xs => hash (3, (is.map Fin.val, xs.hashExpr))
  | .apply f x => hash (4, (f.hashExpr, x.hashExpr))
  | .bind op fs ins =>
    hash (5, (toString op, (List.ofFn fun i => (fs i).hashExpr, ins.hashExpr)))

def Expr.addVars {args outs : List data} (expr : Expr op args outs) (code : String) :
    CodeM (List Nat) :=
  let n_new_var : Nat := outs.length
  fun ⟨n_var, codes, libs⟩ ↦
    let new_var_ids := List.ofFn fun (i : Fin n_new_var) ↦ n_var + i.val
    ⟨new_var_ids, n_var + n_new_var, codes.concat ⟨expr.hashExpr, new_var_ids, code⟩, libs⟩

def Expr.processCode (out_names : List String) (codes : Cached (List Nat × String)) : String :=
  let body : String := "\n".intercalate <|
    codes.map fun ⟨_, assign_id, line⟩ =>
      let assign_names := ",".intercalate (assign_id.map fun i => s!"%{i}")
      s!"{assign_names} = {line}"
  s!"{body}\nreturn {",".intercalate out_names}"

mutual

partial def Expr.addLib {args outs : List data} (expr : Expr op args outs) : CodeM Nat :=
  fun ⟨n_var, codes, libs⟩ ↦
    match libs.findIdx? (fun ⟨addr, _⟩ ↦ expr.hashExpr == addr) with
    | none =>
      let ⟨out_names, _, expr_codes, libs⟩ := expr.genCode ⟨0, [], libs⟩
      let expr_code := processCode out_names expr_codes
      ⟨libs.length, n_var, codes, libs.concat ⟨expr.hashExpr, expr_code⟩⟩
    | some i =>
      ⟨i, n_var, codes, libs⟩

partial def Expr.genCode {args outs : List data} (expr : Expr op args outs) :
    CodeM (List String) := do
  let ⟨_, codes, _⟩ ← get
  match codes.find? (fun ⟨addr, _⟩ ↦ expr.hashExpr == addr) with
  | none => expr.genCode'
  | some ⟨_, out_ids, _⟩ =>
    return out_ids.map fun n => s!"%{n}"

/-- Body of `Expr.genCode`, split out so that the cases are handled by the
equation compiler: a `match` on `expr` inside a `do` block cannot unify the
indices of `Expr`. -/
partial def Expr.genCode' {args outs : List data} :
    Expr op args outs → CodeM (List String)
  | .nil => return []
  | .arg i => return [s!"${i}"]
  | .append xs ys => do
    let xs ← xs.genCode
    let ys ← ys.genCode
    return xs ++ ys
  | .select is xs => do
    let xs ← xs.genCode
    return is.map fun i =>
      match xs[i]? with
      | none => ""
      | some a => a
  | .bind op exprs ins => do
    let expr := Expr.bind op exprs ins
    let exprs ← (List.ofFn fun i => (exprs i).addLib).mapM id
    let exprs := exprs.map fun i => s!"@{i}"
    let ins ← ins.genCode
    let out_ids ← expr.addVars s!"{op}; {", ".intercalate (exprs ++ ins)}"
    return out_ids.map fun n ↦ s!"%{n}"
  | .apply f x => do
    let expr := Expr.apply f x
    let f ← f.addLib
    let x ← x.genCode
    let out_ids ← expr.addVars s!"call; {", ".intercalate (s!"@{f}" :: x)}"
    return out_ids.map fun n ↦ s!"%{n}"

end

def Expr.code {args outs : List data} (expr : Expr op args outs) : String :=
  let ⟨out_names, _, codes, libs⟩ := expr.genCode ⟨0, [], []⟩
  let body := processCode out_names codes
  let libs := "\n\n".intercalate <| List.ofFn fun (i : Fin libs.length) => s!"@{i}:\n{libs[i].2}"
  s!"{body}\n\n{libs}"

@[reduce_soir]
def Expr.ofFn {args outs : List data}
  (f : Curry (Expr op args [·]) args (Expr op args outs))
  : Expr op args outs :=
  f.get fun i => Expr.arg i

end

def evalType {data : Type} (impl : data → Type) (args outs : List data) : Type :=
  Curry impl args (Index impl outs)

def Impl.bindType {data : Type} (impl : data → Type)
  (exprs : List (List data × List data)) (args outs : List data) :=
  match exprs with
  | [] => evalType impl args outs
  | expr :: exprs => evalType impl expr.1 expr.2 → bindType impl exprs args outs

/-- Logical implementation of the op -/
class Impl {data : Type} (op : OpType data) (impl : data → Type) where
  bind {expr : List (List data × List data)} {args outs : List data} : 
    op expr args outs → Impl.bindType impl expr args outs

@[reduce_soir]
def evalType.bind {data : Type} {impl : data → Type} {exprs : List (List data × List data)}
  {args outs : List data} :
    Impl.bindType impl exprs args outs →
    (∀ i : Fin exprs.length, evalType impl exprs[i].1 exprs[i].2) → evalType impl args outs :=
  match exprs with
  | [] => fun x _ => x
  | expr :: exprs => fun op fs => bind (op (fs ⟨0, by simp⟩)) (fun i => fs i.succ)

/-- We can evaluate an expression using some implementation -/
@[reduce_soir]
def Expr.eval {data : Type} {opType : OpType data} {args outs : List data}
  (impl : data → Type) [Impl opType impl] : Expr opType args outs → evalType impl args outs
  | nil => Curry.pure Index.null
  | arg i => (Curry.arg i).map Index.single
  | append x y =>  Curry.map₂ Index.append (x.eval impl) (y.eval impl)
  | select i x => (x.eval impl).map (Index.select i)
  | bind op fs xs =>
    let op := Impl.bind (impl := impl) op
    let op := evalType.bind op fun i => (fs i).eval impl
    (xs.eval impl).map op.get
  | apply f xs => (xs.eval impl).map (f.eval impl).get

inductive SimpleOp {data : Type} (op : List data → data → Type) : OpType data where
  | simple {args : List data} {out : data} : op args out → SimpleOp op [] args [out]

class SimpleImpl {data : Type} (op : List data → data → Type) (impl : data → Type) where
  bind {args : List data} {out : data} : op args out → Curry impl args (impl out)

attribute [reduce_soir] Impl.bind SimpleImpl.bind

@[reduce_soir]
instance SimpleOp.instImpl {data : Type} (op : List data → data → Type) (impl : data → Type)
  [SimpleImpl op impl] : Impl (SimpleOp op) impl where
  bind op :=
    match op with
    | simple op => (SimpleImpl.bind op).map Index.single

@[reduce_soir]
instance SimpleOp.instToString {data : Type} (op : List data → data → Type)
  [∀ args, ∀ outs, ToString (op args outs)] (exprs : List (List data × List data))
  (args outs : List data) : ToString (SimpleOp op exprs args outs) where
    toString op := match op with
    | .simple op => toString op

inductive CombineOp {data : Type} (op₀ op₁ : OpType data) : OpType data where
  | left {exprs : List (List data × List data)} {args outs : List data} :
    op₀ exprs args outs → CombineOp op₀ op₁ exprs args outs
  | right {exprs : List (List data × List data)} {args outs : List data} :
    op₁ exprs args outs → CombineOp op₀ op₁ exprs args outs

@[reduce_soir]
instance CombineOp.instImpl {data : Type} (op₀ op₁ : OpType data) (impl : data → Type)
  [Impl op₀ impl] [Impl op₁ impl] : Impl (CombineOp op₀ op₁) impl where
  bind
  | .left op
  | .right op => Impl.bind op

@[reduce_soir]
instance CombineOp.instToString {data : Type} {op₀ op₁ : OpType data}
  [∀ exprs, ∀ args outs, ToString (op₀ exprs args outs)]
  [∀ exprs, ∀ args outs, ToString (op₁ exprs args outs)] :
  ∀ exprs, ∀ args outs, ToString ((CombineOp op₀ op₁) exprs args outs) :=
  fun _ _ _ => {
    toString x := match x with | .left op | .right op => toString op
  }

@[reduce_soir]
def Expr.join {data : Type} {op : OpType data} {args outs : List data} :
    Curry (fun α ↦ Expr op args [α]) outs (Expr op args outs) :=
  match outs with
  | [] => .nil
  | _ :: _ => fun x => Curry.of <| fun xs => x.append <| join.get xs

def Expr.inlineApply {data : Type} {op : OpType data} {args ins outs : List data} :
    Expr op ins outs → Expr op args ins → Expr op args outs
  | .nil, _ => .nil
  | .arg i, x => x.select [i]
  | .append a b, x => (a.inlineApply x).append (b.inlineApply x)
  | .apply f a, x => .apply f (a.inlineApply x)
  | .select i a, x => (a.inlineApply x).select i
  | .bind op fs as, x => .bind op fs (as.inlineApply x)

def Expr.succ {data : Type} {op : OpType data} {a : data} {b c : List data} :
    Expr op b c → Expr op (a :: b) c
  | .nil => .nil
  | .arg ⟨i, hi⟩ => .arg <| .mk (i + 1) <| by simpa
  | .append a b => a.succ.append b.succ
  | .apply f a => .apply f a.succ
  | .select i a => a.succ.select i
  | .bind op fs as => .bind op fs as.succ

def Expr.ofAppend {data : Type} {op : OpType data} {a b c : List data} :
    Expr op a b → Expr op (c ++ a) b :=
  match c with
  | [] => id
  | _ :: _ => fun x ↦ x.ofAppend.succ

def Expr.cast_arg {data : Type} {op : OpType data} {a b : List data} (i : Fin a.length) :
    Expr op (a ++ b) [a[i]] :=
  match a with
  | a₀ :: a =>
    match i with
    | .mk 0 _ => .arg <| .mk 0 <| by simp
    | .mk (i + 1) h => succ <| cast_arg <| .mk i <| by simpa using h

def Expr.ofAppend' {data : Type} {op : OpType data} {a b c : List data} :
    Expr op a b → Expr op (a ++ c) b
  | .arg i => cast_arg i
  | .nil => .nil
  | .apply f a => .apply f a.ofAppend'
  | .bind op fs as => .bind op fs as.ofAppend'
  | .append a b => a.ofAppend'.append b.ofAppend'
  | .select i a => a.ofAppend'.select i

def Expr.id {data : Type} {op : OpType data} {args : List data} :
    Expr op args args :=
  match args with
  | [] => .nil
  | a :: args =>
    .append (.arg ⟨0, by simp⟩) <| Expr.id.succ

end Soir
