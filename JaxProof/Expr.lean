import JaxProof.Tensor
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.ENat.Defs

/-!
This file contains the inductive type `JAX.Expr`, which represents a jaxpr, and `JAX.Expr.code`
which generates the python code.
-/

namespace Jax

inductive DType where
  | float : DType
  | int : DType

abbrev Shape : Type := List ℕ

structure TensorType where
  dtype : DType
  shape : List ℕ

instance : ToString DType where
  toString x := match x with | .float => "float" | .int => "int"

/-
inductive DList {α : Type} (m : α → Type) : List α → Type where
  | nil : DList m []
  | cons {x : α} {xs : List α} : m x → DList m xs → DList m (x :: xs)

syntax "*[" term,* "]" : term
open Lean in macro_rules
  | `(term| *[ $[$items],* ]) =>
    let rec parse : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(DList.nil)
    | x :: xs => do
      let xs ← parse xs
      `(DList.cons $x $xs) 
    let ⟨items⟩ := items
    parse items

def DList.mapM {α β : Type} {γ : α → Type} {x : List α} {m : Type → Type} [Monad m]
  (f : {x : α} → γ x → m β) : DList γ x → m (List β)
  | nil => pure []
  | cons a as => do
    let a ← f a
    let as ← as.mapM f
    return a :: as

def DList.get {α : Type} {γ : α → Type} {a : List α}
    (i : Fin a.length) (x : DList γ a) : γ a[i] :=
  match a with
  | a :: as =>
    match x with
    | cons x xs =>
      match i with
      | .mk 0 _ => x
      | .mk (n + 1) h => xs.get <| .mk n <| by simpa using h

def DList.of {α : Type} {γ : α → Type} {a : List α} (x : ∀ i : Fin a.length, γ a[i]) : DList γ a :=
  match a with
  | [] => *[]
  | _ :: _ => .cons (x 0) <| DList.of fun i => x i.succ
-/

inductive Expr (opType : List TensorType → TensorType → Type) :
    List TensorType → List TensorType → Type where
  | arg {args : List TensorType} (i : Fin args.length) : Expr opType args [args[i]]
  | bind {args : List TensorType} {ins : List TensorType} {out : TensorType} :
    opType ins out → Expr opType args ins → Expr opType args [out]
  | apply {args ins outs: List TensorType} :
    Expr opType ins outs → Expr opType args ins  → Expr opType args outs
  | append {args outs outs' : List TensorType} :
    Expr opType args outs → Expr opType args outs' → Expr opType args (outs ++ outs')
  | select {args outs : List TensorType} (i : List (Fin outs.length)) :
    Expr opType args outs → Expr opType args (i.map outs.get)

variable {opType : List TensorType → TensorType → Type} [∀ args, ∀ out, ToString (opType args out)]

abbrev Cached (α : Type) : Type := List (USize × α)

unsafe def Expr.addLine {args outs : List TensorType}
  (expr : Expr opType args outs) (code : String) :
    StateM (ℕ × Cached (List ℕ × String) × Cached String) (List ℕ) :=
  fun ⟨n_var, codes, libs⟩ ↦
    match codes.find? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let out_ids : List ℕ := List.ofFn fun (i : Fin outs.length) ↦ i.val + n_var
      let new_codes := codes.concat ⟨ptrAddrUnsafe expr, out_ids, code⟩
      ⟨out_ids, outs.length + n_var, new_codes, libs⟩
    | some ⟨_, out_ids, _⟩ =>
      ⟨out_ids, n_var, codes, libs⟩

unsafe def Expr.addLib {args outs : List TensorType}
  (expr : Expr opType args outs) (code : String) :
    StateM (ℕ × Cached (List ℕ × String) × Cached String) ℕ :=
  fun ⟨n_var, codes, libs⟩ ↦
    match libs.findIdx? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      ⟨libs.length, n_var, codes, libs.concat ⟨ptrAddrUnsafe expr, code⟩⟩
    | some i =>
      ⟨i, n_var, codes, libs⟩

def Expr.processCode (out_names : List String) (codes : Cached (List ℕ × String)) : String :=
  let body : String := "\n".intercalate <|
    codes.map fun ⟨_, assign_id, line⟩ =>
      let assign_names := ",".intercalate (assign_id.map fun i => s!"%{i}")
      s!"{assign_names} = {line}"
  s!"{body}\nreturn {",".intercalate out_names}"

unsafe def Expr.genCode {args outs : List TensorType} (expr : Expr opType args outs) :
    StateM (ℕ × Cached (List ℕ × String) × Cached String) (List String) :=
  match expr with
  | arg i => pure [s!"${i}"]
  | bind op xs => do
    let xs ← xs.genCode
    let out_id ← expr.addLine s!"{op} {",".intercalate xs}"
    return out_id.map fun n ↦ s!"%{n}"
  | apply fn xs => do
    let ⟨n_var, codes, libs⟩ ← get
    let ⟨out_names, _, fn_codes, libs⟩ := fn.genCode ⟨0, [], libs⟩
    set (n_var, codes, libs)
    let fn_code := processCode out_names fn_codes
    let fn_id ← fn.addLib fn_code
    let xs ← xs.genCode
    let out_id ← expr.addLine s!"call @{fn_id} {",".intercalate xs}"
    return out_id.map fun n ↦ s!"%{n}"
  | append xs ys => do
    let xs ← xs.genCode
    let ys ← ys.genCode
    return xs ++ ys
  | select is xs => do
    let xs ← xs.genCode
    return is.map fun i =>
      match xs[i]? with
      | none => ""
      | some a => a
    
unsafe def Expr.code {args outs : List TensorType} (expr : Expr opType args outs) : String :=
  let ⟨out_names, _, codes, libs⟩ := expr.genCode ⟨0, [], []⟩
  let body := processCode out_names codes
  let libs := "\n".intercalate <| List.ofFn fun (i : Fin libs.length) => s!"@{i}\n{libs[i]}"
  s!"{body}\n{libs}"

declare_syntax_cat expr_builder

/--
Define an XLA expression. Example:
xla_fun foobar (n m l k : ℕ)
arguments
  a : float [n,m],
  b : float [n,l],
  c : float [n,k]
returns
  float [n,m + l + k],
  float [n,m + l]
begin
  let_expr d : float [n,m + l] := .binop (.concat (batch:=[n]) (axis:=1) (n:=m) (m:=l)) a b;
  let_expr d' : float [n,m + l] := .unop .cos d;
  return .binop (.concat (batch:=[n]) (axis:=1) (n:=m + l) (m:=k)) d' c, d
-/
syntax "define_expr" "using" term "with" ( ident ":" term ),*
       "begin" expr_builder : term

/--
Custom `let` binder for XLA expressions 
-/
syntax "let_expr" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":=" term ";" expr_builder : expr_builder
syntax "return" term : expr_builder
--syntax "fori_loop" term "," term "," term "," term : expr_builder

open Lean in
partial def parse_expr_builder (opType args : TSyntax `term) :
    TSyntax `expr_builder → MacroM (TSyntax `term)
  | `(expr_builder| let_expr $name : $outs := $val; $content) => do
    `(term| let $name : Expr $opType $args $outs := $val;
            $(← parse_expr_builder opType args content))
  | `(expr_builder| let $name : $type := $val; $content) => do
    `(term| let $name : $type := $val; $(← parse_expr_builder opType args content))
  | `(expr_builder| let $name := $val; $content) => do
    `(term| let $name := $val; $(← parse_expr_builder opType args content))
  | `(expr_builder| return $rets) => pure rets
--  | `(expr_builder| fori_loop $n, $fn , $init , $aux) =>
--    `(term| ExprGroup.fori_loop $n $fn $init $aux)
  | _ => Macro.throwUnsupported

open Lean in macro_rules
  | `(define_expr using $opType with $[$argnames : $argtypes],* begin $body) => do
    let args : TSyntax `term ← `(term| [ $[$argtypes],* ])
    let parsed_body : TSyntax `term ← parse_expr_builder opType args body
    let rec bind_args : List (TSyntax `ident × TSyntax `term) → ℕ → MacroM (TSyntax `term)
      | [], _ => return parsed_body
      | ⟨name, out⟩ :: rest, n => do
        `(term| let $name : Expr $opType $args [$out] := Expr.arg $(quote n);
        $(← bind_args rest (n + 1)))
    bind_args (argnames.toList.zip argtypes.toList) 0

inductive SimpleOp : List TensorType → TensorType → Type where
  | iota (n : ℕ) : SimpleOp [] ⟨.float, [n]⟩
  | add {σ : TensorType} : SimpleOp [σ, σ] σ

def SimpleOp.expr_add {args : List TensorType} {σ : TensorType} (x y : Expr SimpleOp args [σ]) :
    Expr SimpleOp args [σ] :=
  let feedin : Expr SimpleOp args [σ, σ] := Expr.append x y
  Expr.bind SimpleOp.add feedin

def foobar (n : ℕ) :=
  define_expr using SimpleOp with
    x : ⟨.float, [n]⟩
  begin
    return SimpleOp.expr_add x x

#print foobar

/-
open Lean in macro_rules
  | `(xla with $[$argnames : $argdtypes $argshapes],*
      returns $[$retdtypes $retshapes],*
      begin $content) => do
    let arglist : TSyntax `term ← `(term| [ $[⟨.$argdtypes, $argshapes⟩],* ])
    let retlist : TSyntax `term ← `(term| [ $[⟨.$retdtypes, $retshapes⟩],* ])
    let argId : Array (TSyntax `term) := Array.ofFn fun (i : Fin argnames.size) => quote i.val
    let parsed : TSyntax `term ← parse_expr_builder arglist content
    let args : List (TSyntax `ident × TSyntax `ident × TSyntax `term × TSyntax `term) :=
      Array.toList <| argnames.zip <| argdtypes.zip <| argshapes.zip <| argId
    let rec bind_args :
      List (TSyntax `ident × TSyntax `ident × TSyntax `term × TSyntax `term)
      → MacroM (TSyntax `term)
    | [] => return parsed
    | ⟨name, dtype, shape, id⟩ :: args =>
      do `(let $name : Expr $arglist ⟨.$dtype, $shape⟩ := .arg $id; $(← bind_args args))
    `(term| ( $(← bind_args args) : ExprGroup $arglist $retlist) )
-/

end Jax
