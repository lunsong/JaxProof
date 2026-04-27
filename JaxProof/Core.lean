import JaxProof.Curry

/-!
# The Core of the SSA framework

This file contains definitions for code generation and native evaluation
-/

namespace SSA

/-- `OpType` specifies the primitive ops. First-order and second-order ops are put together. -/
def OpType (data : Type) : Type 1 := List (List data × List data) → List data → List data → Type

/-- `Expr` represent an SSA expression with multiput input and multiple output, using `data`
as the data type and `op` as primitive ops. -/
inductive Expr {data : Type} (op : OpType data) :
    List data → List data → Type where
  | append {args outs outs' : List data} :
    Expr op args outs → Expr op args outs' → Expr op args (outs ++ outs')
  | select {args outs : List data} (i : List (Fin outs.length)) :
    Expr op args outs → Expr op args (i.map outs.get)
  | arg {args : List data} (i : Fin args.length) : Expr op args [args[i]]
  | bind {args ins outs : List data} {exprs : List (List data × List data)} :
    op exprs ins outs →
      (∀ i : Fin exprs.length, Expr op exprs[i].1 exprs[i].2) → Expr op args ins → Expr op args outs

variable {data : Type} {op : OpType data} [∀ exprs, ∀ ins, ∀ outs, ToString (op exprs ins outs)]

abbrev Cached (α : Type) : Type := List (USize × α)

abbrev Expr.CodeM (α : Type) : Type :=
  StateM (Nat × Cached (List Nat × String) × Cached String) α

unsafe def Expr.addVars {args outs : List data} (expr : Expr op args outs) (code : String) :
    CodeM (List Nat) :=
  let n_new_var : Nat := outs.length
  fun ⟨n_var, codes, libs⟩ ↦
    let new_var_ids := List.ofFn fun (i : Fin n_new_var) ↦ n_var + i.val
    ⟨new_var_ids, n_var + n_new_var, codes.concat ⟨ptrAddrUnsafe expr, new_var_ids, code⟩, libs⟩

def Expr.processCode (out_names : List String) (codes : Cached (List Nat × String)) : String :=
  let body : String := "\n".intercalate <|
    codes.map fun ⟨_, assign_id, line⟩ =>
      let assign_names := ",".intercalate (assign_id.map fun i => s!"%{i}")
      s!"{assign_names} = {line}"
  s!"{body}\nreturn {",".intercalate out_names}"

mutual

unsafe def Expr.addLib {args outs : List data} (expr : Expr op args outs) : CodeM Nat :=
  fun ⟨n_var, codes, libs⟩ ↦
    match libs.findIdx? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let ⟨out_names, _, expr_codes, libs⟩ := expr.genCode ⟨0, [], libs⟩
      let expr_code := processCode out_names expr_codes
      ⟨libs.length, n_var, codes, libs.concat ⟨ptrAddrUnsafe expr, expr_code⟩⟩
    | some i =>
      ⟨i, n_var, codes, libs⟩

unsafe def Expr.genCode {args outs : List data} (expr : Expr op args outs) :
    CodeM (List String) := do
  let ⟨_, codes, _⟩ ← get
  match codes.find? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
  | none =>
    match expr with
    | arg i => return [s!"${i}"]
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
    | bind op exprs ins =>
      let exprs ← (List.ofFn fun i => (exprs i).addLib).mapM id
      let exprs := ",".intercalate (exprs.map fun i => s!"@{i}")
      let ins ← ins.genCode
      let out_ids ← expr.addVars s!"{op} {exprs},{",".intercalate ins}"
      return out_ids.map fun n ↦ s!"%{n}"
      
  | some ⟨_, out_ids, _⟩ =>
    return out_ids.map fun n => s!"%{n}"

end

unsafe def Expr.code {args outs : List data} (expr : Expr op args outs) : String :=
  let ⟨out_names, _, codes, libs⟩ := expr.genCode ⟨0, [], []⟩
  let body := processCode out_names codes
  let libs := "\n\n".intercalate <| List.ofFn fun (i : Fin libs.length) => s!"@{i}\n{libs[i].2}"
  s!"{body}\n\n{libs}"

declare_syntax_cat expr_builder

syntax "ssa" term "with" ( ident ":" term ),*
       "begin" expr_builder : term

syntax "let_expr" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":=" term ";" expr_builder : expr_builder
syntax "return" term : expr_builder

open Lean in
partial def parse_expr_builder (expr : TSyntax `term) :
    TSyntax `expr_builder → MacroM (TSyntax `term)
  | `(expr_builder| let_expr $name : $outs := $val; $content) => do
    `(term| let $name : $expr $outs := $val;
            $(← parse_expr_builder expr content))
  | `(expr_builder| let $name : $type := $val; $content) => do
    `(term| let $name : $type := $val; $(← parse_expr_builder expr content))
  | `(expr_builder| let $name := $val; $content) => do
    `(term| let $name := $val; $(← parse_expr_builder expr content))
  | `(expr_builder| return $rets) => pure rets
  | _ => Macro.throwUnsupported

open Lean in macro_rules
  | `(ssa $op with $[$argnames : $argtypes],* begin $body) => do
    let args : TSyntax `term ← `(term| [ $[$argtypes],* ])
    let expr_head : TSyntax `term ← `(term| Expr $op $args)
    let parsed_body : TSyntax `term ← parse_expr_builder expr_head body
    let rec bind_args : List (TSyntax `ident × TSyntax `term) → Nat → MacroM (TSyntax `term)
      | [], _ => return parsed_body
      | ⟨name, out⟩ :: rest, n => do
        `(term| let $name : $expr_head [$out] := Expr.arg ⟨$(quote n), by decide⟩;
        $(← bind_args rest (n + 1)))
    bind_args (argnames.toList.zip argtypes.toList) 0

def Impl.bindType_simple {data : Type} (impl : data → Type) (args outs : List data) : Type :=
  Curry (args.map impl) (Index (outs.map impl))

def Impl.bindType {data : Type} (impl : data → Type)
  (exprs : List (List data × List data)) (args outs : List data) :=
  Curry (exprs.map fun ⟨a, b⟩ => bindType_simple impl a b) (bindType_simple impl args outs)

class Impl (data : Type) (impl : data → Type) (op : OpType data) where
  bind (expr : List (List data × List data)) (args outs : List data) : 
    op expr args outs → Impl.bindType impl expr args outs

inductive SimpleOp (data : Type) : List (List data × List data) → List data → List data → Type
  | call {α β : List data} : SimpleOp data [(α, β)] α β
  | node {α : data} : SimpleOp data [] [α, α] [α]

instance (exprs : List (List String × List String)) (args outs : List String) :
    ToString (SimpleOp String exprs args outs) where
  toString op :=
    match op with
    | .call => "call"
    | .node => "node"

def foobar :=
  ssa SimpleOp String with x : "Float", y : "Float" begin
  let_expr z : ["Float"] := .bind .node (fun i => nomatch i) (.append x y);
  return z

def barfoo :=
  ssa SimpleOp String with x : "Float", y : "Float" begin
  let_expr z : ["Float"] := .bind .call (fun i => match i with | .mk 0 _ => foobar) (x.append y);
  return z

#eval IO.println barfoo.code

end SSA


