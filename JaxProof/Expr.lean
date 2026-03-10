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

namespace _dot_general

def get_shape (batch contract indep : List ℕ)
  (i : List (Fin batch.length ⊕ Fin contract.length ⊕ Fin indep.length)) : List ℕ :=
  i.map fun i => match i with
  | .inl i => batch[i]
  | .inr (.inl i) => contract[i]
  | .inr (.inr i) => indep[i]

end _dot_general

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
  | dot_general {α : DType}
    (batch contract lhs_indep rhs_indep: List ℕ)
    (lhs : List (Fin batch.length ⊕ Fin contract.length ⊕ Fin lhs_indep.length))
    (rhs : List (Fin batch.length ⊕ Fin contract.length ⊕ Fin rhs_indep.length))
    (lhs_inj : Function.Injective lhs.get := by decide)
    (lhs_surj : Function.Surjective lhs.get := by decide)
    (rhs_inj : Function.Injective rhs.get := by decide)
    (rhs_surj : Function.Surjective rhs.get := by decide)
    : 
    Op [⟨α, _dot_general.get_shape batch contract lhs_indep lhs⟩,
        ⟨α, _dot_general.get_shape batch contract rhs_indep rhs⟩]
      ⟨α, batch ++ lhs_indep ++ rhs_indep⟩
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
  | mul => "mul"
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

inductive ExprGroup : List TensorType → List TensorType → Type where
  | nil {args : List TensorType} : ExprGroup args []
  | cons {args : List TensorType} {x : TensorType} {xs : List TensorType} :
    Expr args x → ExprGroup args xs → ExprGroup args (x :: xs)
  | append {args outs outs' : List TensorType} : 
    ExprGroup args outs → ExprGroup args outs' → ExprGroup args (outs ++ outs')
  | apply {xs ys zs : List TensorType} :
    ExprGroup xs ys → ExprGroup ys zs → ExprGroup xs zs
  | fori_loop {args carry aux : List TensorType} (n : ℕ) :
    ExprGroup (⟨.int, []⟩ :: carry ++ aux) carry
      → ExprGroup args carry → ExprGroup args aux → ExprGroup args carry

abbrev Cached (α : Type) : Type := List (USize × α)

unsafe def Expr.insert {args : List TensorType} {out : TensorType}
  (expr : Expr args out) (code : String) :
    StateM (Cached String) ℕ := fun ctx ↦
  match ctx.findIdx? (fun x ↦ ptrAddrUnsafe expr == x.1) with
  | none => ⟨ctx.length, ctx.concat ⟨ptrAddrUnsafe expr, code⟩⟩
  | some n => ⟨n, ctx⟩

unsafe def Expr.genCode {args : List TensorType} {out : TensorType}
    (expr : Expr args out) : StateM (Cached String) String :=
  match expr with
  | nullop op => do return "%" ++ toString (← expr.insert op.toString)
  | unop op x => do return "%" ++ toString (← expr.insert s!"{op} {← x.genCode}")
  | binop op x y => do return "%" ++ toString (← expr.insert s!"{op} {← x.genCode} {← y.genCode}")
  | arg i => pure s!"${i}"

def complete_code (commands : Cached String) (outs : String) : String :=
  "\n".intercalate (commands.map Prod.snd) ++ "\nreturn " ++ outs

mutual

unsafe def ExprGroup.insert {args outs : List TensorType}
  (expr : ExprGroup args outs) : StateM (Cached String × Cached String) ℕ :=
  fun ⟨commands, libs⟩ ↦
    match libs.findIdx? (fun x ↦ x.1 == ptrAddrUnsafe expr) with
    | some n => ⟨n, commands, libs⟩
    | none =>
      let ⟨expr_outs, expr_commands, libs⟩ := expr.genCode ⟨[], libs⟩
      ⟨libs.length, commands,
       libs.concat ⟨ptrAddrUnsafe expr, complete_code expr_commands expr_outs⟩⟩


unsafe def ExprGroup.genCode {args outs : List TensorType} :
    ExprGroup args outs → StateM (Cached String × Cached String) String
  | nil => pure ""
  | cons x xs => fun ⟨commands, libs⟩ ↦
    let ⟨x, commands⟩ := x.genCode commands;
    let ⟨xs, commands, libs⟩ := xs.genCode ⟨commands, libs⟩
    ⟨s!"{x}, {xs}", commands, libs⟩
  | append x y => do return s!"{← x.genCode}, {← y.genCode}"
  | apply x f =>
    do return s!"apply(@{← f.insert}, {← x.genCode})"
  | fori_loop n step_fn init aux =>
    do return s!"fori_loop({n}, @{← step_fn.insert}, ({← init.genCode}), ({← aux.genCode}))"

end

unsafe def ExprGroup.code {args outs : List TensorType} : ExprGroup args outs → String :=
  fun expr ↦
    let ⟨outs, commands, libs⟩ := expr.genCode ⟨[], []⟩
    let main := complete_code commands outs
    main ++ "\nwith\n" ++ "\n\n".intercalate (libs.map Prod.snd)

def Exprs (args : List TensorType) : List TensorType → Type
  | [] => Unit
  | σ :: σs => Expr args σ × Exprs args σs

def Exprs.toExprGroup {args outs : List TensorType} : Exprs args outs → ExprGroup args outs :=
  match outs with
  | [] => fun _ ↦ .nil
  | _ :: _ => fun ⟨x, xs⟩ ↦ .cons x xs.toExprGroup

/-
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
-/

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
syntax "xla_fun" ident ("(" ident* ":" term ")")*
       "arguments" ( ident ":" ident term ),*
       "returns" (ident term),*
       "begin" expr_builder : command

/--
Custom `let` binder for XLA expressions 
-/
syntax "let_expr" ident ":" ident term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":=" term ";" expr_builder : expr_builder
syntax "return" term,* : expr_builder

open Lean in macro_rules
  | `(xla_fun $funName $[($params* : $paramtype)]*
      arguments $[$argnames : $argdtypes $argshapes],*
      returns $[$retdtypes $retshapes],*
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
    | `(expr_builder| return $rets,*) => do
      `(term| ( ⟨ $rets,* , () ⟩ : Exprs $arglist $retlist))
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
    --`(def $funName : Exprs $arglist $retlist := $(← bind_args args))
    `(def $funName $[($params* : $paramtype)]* : ExprGroup $arglist $retlist :=
      Exprs.toExprGroup <| $(← bind_args args))

end Jax
