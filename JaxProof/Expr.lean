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

inductive Op : List TensorType → TensorType → Type where
  | abs {σ : TensorType} : Op [σ] σ
  | acos {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | acosh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | add {σ : TensorType} : Op [σ, σ] σ
  | and {s : Shape} : Op [⟨.int, s⟩, ⟨.int, s⟩] ⟨.int, s⟩
  | argmax {σ : TensorType} (axis : ℕ) : Op [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | argmin {σ : TensorType} (axis : ℕ) : Op [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | asin {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | asinh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | atan {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | atanh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i0e {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i1e {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | broadcast {α : DType} (s : List (ℕ × Bool)) :
    Op [⟨α, Tensor.preBroadcast s⟩] ⟨α, s.map Prod.fst⟩
  | cbrt {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | ceil {s : Shape} : Op [⟨.float, s⟩] ⟨.int, s⟩
  | cholesky {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | concat {α : DType} {batch : Shape} {n m : ℕ} {axis : ℕ} :
    Op [⟨α, batch.insertIdx axis n⟩, ⟨α, batch.insertIdx axis m⟩] ⟨α, batch.insertIdx axis (n + m)⟩
  | conv {α : DType} {s : Shape} {n m : ℕ} {axis : ℕ} :
    Op [⟨α, s.insertIdx axis n⟩, ⟨α, s.insertIdx axis m⟩] ⟨α, s.insertIdx axis (n + m)⟩
  | convert_type {α β : DType} {s : Shape} : Op [⟨α, s⟩] ⟨β, s⟩
  | cos {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | cosh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | cumlogsumexp {s : Shape} (axis : ℕ) (reverse : Bool) : Op [⟨.float, s⟩] ⟨.float, s⟩
  | cummax {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | cummin {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | cumprod {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | cumsum {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | div {σ : TensorType} : Op [σ, σ] σ
  | dot_general {α : DType} (batch contract lhs rhs : Shape) :
    Op [⟨α, batch ++ contract ++ lhs⟩, ⟨α, batch ++ contract ++ rhs⟩] ⟨α, batch ++ lhs ++ rhs⟩
  | dynamic_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    Op [⟨α, dims.map Prod.fst⟩] ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩
  | dynamic_update_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    Op [⟨α, dims.map Prod.fst⟩, ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩] ⟨α, dims.map Prod.fst⟩
  | eigvals {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvalsh {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvecs {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | eigvecsh {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | empty {σ : TensorType} : Op [] σ
  | eq {σ : TensorType} : Op [σ, σ] ⟨.int, σ.shape⟩
  | erf {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | erf_inv {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | erfc {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | exp {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | exp2 {σ : TensorType} : Op [σ] σ
  | expm1 {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  --| fft {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  --| gather {α : DType} {s s' batch: Shape} : Op [⟨α,⟩
  | iota (n : ℕ) : Op [] ⟨.int, [n]⟩
  | mul {σ : TensorType} : Op [σ, σ] σ
  | mod {σ : TensorType} : Op [σ, σ] σ
  | div_int {σ : TensorType} : Op [σ, σ] σ
  --| neg {σ : TensorType} : Op [σ] σ
  --| lt : Op (some 2)
  --| select : Op (some 3)
  --| addIdx : Op (some 3)
  --| sin : Op (some 1)
  --| log : Op (some 1)
  --| sqrt : Op (some 1)
  --| einsum (s : List ℕ) : List (List (Fin s.length)) → ℕ → Op none
  --| tuple : Op none
  --| tupleGet : ℕ → Op (some 1)
  --| anonTuple : Op none

def Op.reprType : ℕ∞ → Type
  | none => List String → String
  | some n => curryType String n

def Op.toString {args : List TensorType} {out : TensorType} : Op args out → String
  | add => "add"
  | cos => "cos"
  | concat => "concat"
  | _ => "unimplemented"

instance (args : List TensorType) (out : TensorType) : ToString (Op args out) :=
  ⟨Op.toString⟩

def argString {α : Type} [ToString α] (xs : List α) : String :=
  ", ".intercalate (xs.map toString)

def argType (α : TensorType → Type) : List TensorType → Type
  | [] => Unit
  | σ :: σs => α σ × argType α σs

inductive Expr (args : List TensorType) : TensorType → Type where
  | nullop {out : TensorType} : Op [] out → Expr args out
  | unop {x out : TensorType} : Op [x] out → Expr args x → Expr args out
  | binop {x y out : TensorType} : Op [x, y] out → Expr args x → Expr args y → Expr args out
  | arg (i : Fin args.length) : Expr args args[i]

abbrev Exprs (args : List TensorType) : List TensorType → Type
  | [] => Unit
  | σ :: σs => Expr args σ × Exprs args σs

unsafe def Expr.insert {args : List TensorType} {out : TensorType}
  (expr : Expr args out) (code : String) :
    StateM (List (USize × String)) ℕ := fun ctx ↦
  match ctx.findIdx? (fun x ↦ ptrAddrUnsafe expr == x.1) with
  | none => ⟨ctx.length, ctx.concat ⟨ptrAddrUnsafe expr, code⟩⟩
  | some n => ⟨n, ctx⟩

unsafe def Expr.genCode {args : List TensorType} {out : TensorType}
    (expr : Expr args out) : StateM (List (USize × String)) String :=
  match expr with
  | nullop op => do return "%" ++ toString (← expr.insert op.toString)
  | unop op x => do return "%" ++ toString (← expr.insert s!"{op} {← x.genCode}")
  | binop op x y => do return "%" ++ toString (← expr.insert s!"{op} {← x.genCode} {← y.genCode}")
  | arg i => pure s!"${i}"

unsafe def Exprs.genCode {args outs : List TensorType} (exprs : Exprs args outs) :
    StateM (List (USize × String)) String :=
  match outs with
  | [] => pure ""
  | _ :: _ =>
    let ⟨expr, exprs⟩ := exprs
    do return s!"{← expr.genCode} {← exprs.genCode}"

unsafe def Exprs.code {args outs : List TensorType} (exprs : Exprs args outs) :
    String :=
  let ⟨out, codes⟩ := exprs.genCode []
  "\n".intercalate (codes.map Prod.snd) ++ "\nreturn " ++ out


def fn : Expr [⟨.float, [3,3]⟩, ⟨.float, [3,4]⟩] ⟨.float, [3,7]⟩ :=
  .unop .cos <| .binop (.concat (batch:=[3]) (n:=3) (m:=4) (axis:=1)) (.arg 0) (.arg 1)

def fn' : Expr [⟨.float, [3,3]⟩, ⟨.float, [3,4]⟩] ⟨.float, [3,7]⟩ :=
  .binop .add fn fn

#eval IO.println ("\n".intercalate ((fn'.genCode []).2.map Prod.snd))

declare_syntax_cat expr_builder

syntax "xla_fun" ident (ident term),*
       "with" ( ident ":" ident term ),*
       "begin" expr_builder : command
syntax "let_expr" ident ":" ident term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":=" term ";" expr_builder : expr_builder
syntax "return" term,* : expr_builder

open Lean in macro_rules
  | `(xla_fun $funName $[$retdtypes $retshapes],*
      with $[$argnames : $argdtypes $argshapes],*
      begin $content) => do
    let arglist : TSyntax `term ← `(term| [ $[⟨.$argdtypes, $argshapes⟩],* ])
    let retlist : TSyntax `term ← `(term| [ $[⟨.$retdtypes, $retshapes⟩],* ])
    let rec parse : TSyntax `expr_builder → MacroM (TSyntax `term)
    | `(expr_builder| let_expr $name : $dtype $shape := $val; $content) => do
      `(term| let $name : Expr $arglist ⟨.$dtype, $shape⟩ := $val; $(← parse content))
    | `(expr_builder| let $name : $type := $val; $content) => do
      `(term| let $name : $type := $val; $(← parse content))
    | `(expr_builder| let $name := $val; $content) => do
      `(term| let $name := $val; $(← parse content))
    | `(expr_builder| return $rets) => do
      `(term| ( ⟨ $rets , () ⟩ : Exprs $arglist $retlist))
    | _ => Macro.throwUnsupported
    let argId : Array (TSyntax `term) := Array.ofFn fun (i : Fin argnames.size) => quote i.val
    let parsed : TSyntax `term ← parse content
    let args : List (TSyntax `ident × TSyntax `ident × TSyntax `term × TSyntax `term) :=
      Array.toList <| argnames.zip <| argdtypes.zip <| argshapes.zip <| argId
    let rec bind_args :
      List (TSyntax `ident × TSyntax `ident × TSyntax `term × TSyntax `term)
      → MacroM (TSyntax `term)
    | [] => return parsed
    | ⟨name, dtype, shape, id⟩ :: args =>
      do `(let $name : Expr $arglist ⟨.$dtype, $shape⟩ := .arg $id; $(← bind_args args))
    `(def $funName : Exprs $arglist $retlist := $(← bind_args args))


xla_fun foobar
  float [2,3 + 4 + 5]
with
  a : float [2,3],
  b : float [2,4],
  c : float [2,5]
begin
  let_expr d : float [2,3 + 4] := .binop (.concat (batch:=[2]) (axis:=1) (n:=3) (m:=4)) a b;
  let_expr d' : float [2,3 + 4] := .unop .cos d;
  return .binop (.concat (batch:=[2]) (axis:=1) (n:=3 + 4) (m:=5)) d' c

#eval IO.println foobar.code

structure ExprsTuple where
  args : List TensorType
  outs : List TensorType
  fn : Exprs args outs

inductive HiExpr (libs : List ExprsTuple) (args : List TensorType) : List TensorType → Type where
  | arg (i : Fin args.length) : HiExpr libs args [args[i]]
  | concat {α β : List TensorType} :
    HiExpr libs args α → HiExpr libs args β → HiExpr libs args (α ++ β)


end Jax
