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

def TensorType.int_scalar : TensorType := ⟨.int, []⟩

instance : ToString DType where
  toString x := match x with | .float => "float" | .int => "int"

inductive Expr (opType : List TensorType → TensorType → Type) :
    List TensorType → List TensorType → Type where
  | nil {args : List TensorType} : Expr opType args []
  | arg {args : List TensorType} (i : Fin args.length) : Expr opType args [args[i]]
  | bind {args : List TensorType} {ins : List TensorType} {out : TensorType} :
    opType ins out → Expr opType args ins → Expr opType args [out]
  | apply {args ins outs: List TensorType} :
    Expr opType ins outs → Expr opType args ins  → Expr opType args outs
  | append {args outs outs' : List TensorType} :
    Expr opType args outs → Expr opType args outs' → Expr opType args (outs ++ outs')
  | select {args outs : List TensorType} (i : List (Fin outs.length)) :
    Expr opType args outs → Expr opType args (i.map outs.get)
  | fori_loop {args carry : List TensorType} :
    ℕ → Expr opType (TensorType.int_scalar :: carry) carry
      → Expr opType args carry
      → Expr opType args carry

variable {opType : List TensorType → TensorType → Type} [∀ args, ∀ out, ToString (opType args out)]

abbrev Cached (α : Type) : Type := List (USize × α)

abbrev Expr.CodeM (α : Type) : Type :=
  StateM (ℕ × Cached (List ℕ × String) × Cached String) α

unsafe def Expr.addLine {args outs : List TensorType}
  (expr : Expr opType args outs) (code : String) : CodeM (List ℕ) :=
  fun ⟨n_var, codes, libs⟩ ↦
    match codes.find? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let out_ids : List ℕ := List.ofFn fun (i : Fin outs.length) ↦ i.val + n_var
      let new_codes := codes.concat ⟨ptrAddrUnsafe expr, out_ids, code⟩
      ⟨out_ids, outs.length + n_var, new_codes, libs⟩
    | some ⟨_, out_ids, _⟩ =>
      ⟨out_ids, n_var, codes, libs⟩

def Expr.processCode (out_names : List String) (codes : Cached (List ℕ × String)) : String :=
  let body : String := "\n".intercalate <|
    codes.map fun ⟨_, assign_id, line⟩ =>
      let assign_names := ",".intercalate (assign_id.map fun i => s!"%{i}")
      s!"{assign_names} = {line}"
  s!"{body}\nreturn {",".intercalate out_names}"

mutual

unsafe def Expr.addLib {args outs : List TensorType}
  (expr : Expr opType args outs) : CodeM ℕ :=
  fun ⟨n_var, codes, libs⟩ ↦
    match libs.findIdx? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let ⟨out_names, _, fn_codes, libs⟩ := expr.genCode ⟨0, [], libs⟩
      let fn_code := processCode out_names fn_codes
      ⟨libs.length, n_var, codes, libs.concat ⟨ptrAddrUnsafe expr, fn_code⟩⟩
    | some i =>
      ⟨i, n_var, codes, libs⟩

unsafe def Expr.genCode {args outs : List TensorType} (expr : Expr opType args outs) :
    CodeM (List String) :=
  match expr with
  | nil => pure []
  | arg i => pure [s!"${i}"]
  | bind op xs => do
    let xs ← xs.genCode
    let out_id ← expr.addLine s!"{op} {",".intercalate xs}"
    return out_id.map fun n ↦ s!"%{n}"
  | apply fn xs => do
    let fn_id ← fn.addLib
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
  | fori_loop n fn init => do
    let fn ← fn.addLib
    let init ← init.genCode
    let out_id ← expr.addLine s!"fori_loop {n} @{fn} {",".intercalate init}"
    return out_id.map fun n ↦ s!"%{n}"

end

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

instance (args : List TensorType) (out : TensorType) : ToString (SimpleOp args out) where
  toString x :=
    match x with
    | .iota n => s!"iota {n}"
    | .add => s!"add"

def SimpleOp.expr_add {args : List TensorType} {σ : TensorType} (x y : Expr SimpleOp args [σ]) :
    Expr SimpleOp args [σ] :=
  let feedin : Expr SimpleOp args [σ, σ] := Expr.append x y
  Expr.bind SimpleOp.add feedin

def SimpleOp.expr_iota {args : List TensorType} (n : ℕ) :
    Expr SimpleOp args [⟨.float, [n]⟩] :=
  Expr.bind (SimpleOp.iota n) Expr.nil

def foobar (n : ℕ) :=
  define_expr using SimpleOp with
    x : ⟨.float, [n]⟩
  begin
    return SimpleOp.expr_add x (SimpleOp.expr_iota n)

#print foobar

#eval IO.println (foobar 10).code

end Jax
