namespace SSA

/-- 
`Expr` represent and SSA expression with multiput input and multiple output, using `data`
as the data type, `op` as primitive operators and `ho` as supported higher order functions.
-/
inductive Expr {data : Type}
    (op : List data → List data → Type)
    (ho : List (List data × List data) → List data → List data → Type) :
    List data → List data → Type where
  | nil {args : List data} : Expr op ho args []
  | append {args outs outs' : List data} :
    Expr op ho args outs → Expr op ho args outs' → Expr op ho args (outs ++ outs')
  | select {args outs : List data} (i : List (Fin outs.length)) :
    Expr op ho args outs → Expr op ho args (i.map outs.get)
  | arg {args : List data} (i : Fin args.length) : Expr op ho args [args[i]]
  | call {args ins outs: List data} :
    Expr op ho ins outs → Expr op ho args ins  → Expr op ho args outs
  | bindOp {args ins outs: List data} :
    op ins outs → Expr op ho args ins → Expr op ho args outs
  | bindHo {args ins outs : List data} {exprs : List (List data × List data)} :
    ho exprs ins outs →
      (∀ i : Fin exprs.length, Expr op ho exprs[i].1 exprs[i].2) → Expr op ho args ins → Expr op ho args outs

variable {data : Type}
variable {op : List data → List data → Type} [∀ args, ∀ out, ToString (op args out)]
variable {ho : List (List data × List data) → List data → List data → Type}
variable [∀ ins, ∀ args, ∀ outs, ToString (ho ins args outs)]

abbrev Cached (α : Type) : Type := List (USize × α)

abbrev Expr.CodeM (α : Type) : Type :=
  StateM (Nat × Cached (List Nat × String) × Cached String) α

unsafe def Expr.addVars {args outs : List data} (expr : Expr op ho args outs) (code : String) : CodeM (List Nat) :=
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

unsafe def Expr.addLib {args outs : List data} (expr : Expr op ho args outs) : CodeM Nat :=
  fun ⟨n_var, codes, libs⟩ ↦
    match libs.findIdx? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let ⟨out_names, _, expr_codes, libs⟩ := expr.genCode ⟨0, [], libs⟩
      let expr_code := processCode out_names expr_codes
      ⟨libs.length, n_var, codes, libs.concat ⟨ptrAddrUnsafe expr, expr_code⟩⟩
    | some i =>
      ⟨i, n_var, codes, libs⟩

unsafe def Expr.genCode {args outs : List data} (expr : Expr op ho args outs) :
    CodeM (List String) := do
  let ⟨_, codes, _⟩ ← get
  match codes.find? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
  | none =>
    match expr with
    | nil => return []
    | arg i => return [s!"${i}"]
    | bindOp op xs =>
        let xs ← xs.genCode
        let out_id ← expr.addVars s!"{op} {",".intercalate xs}"
        return out_id.map fun n ↦ s!"%{n}"
    | call fn xs => do
      let fn_id ← fn.addLib
      let xs ← xs.genCode
      let out_id ← expr.addVars s!"call @{fn_id} {",".intercalate xs}"
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
    | bindHo ho exprs ins =>
      let exprs ← (List.ofFn fun i => (exprs i).addLib).mapM id
      let exprs := ",".intercalate (exprs.map fun i => s!"@{i}")
      let ins ← ins.genCode
      let out_ids ← expr.addVars s!"{ho} {exprs},{",".intercalate ins}"
      return out_ids.map fun n ↦ s!"%{n}"
      
  | some ⟨_, out_ids, _⟩ =>
    return out_ids.map fun n => s!"%{n}"

end

unsafe def Expr.code {args outs : List data} (expr : Expr op ho args outs) : String :=
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
  | `(ssa $expr with $[$argnames : $argtypes],* begin $body) => do
    let args : TSyntax `term ← `(term| [ $[$argtypes],* ])
    let expr_head : TSyntax `term ← `(term| $expr $args)
    let parsed_body : TSyntax `term ← parse_expr_builder expr_head body
    let rec bind_args : List (TSyntax `ident × TSyntax `term) → Nat → MacroM (TSyntax `term)
      | [], _ => return parsed_body
      | ⟨name, out⟩ :: rest, n => do
        `(term| let $name : $expr_head $args [$out] := Expr.arg $(quote n);
        $(← bind_args rest (n + 1)))
    bind_args (argnames.toList.zip argtypes.toList) 0


inductive SimpleOp : List Nat → List Nat → Type where
  | iota (n : Nat) : SimpleOp [] [n]
  | add {n : Nat} : SimpleOp [n, n] [n] 

inductive SimpleHo : List (List Nat × List Nat) → List Nat → List Nat → Type
  | fori_loop {carry : Nat

def foobar (n : Nat) :=
  ssa Expr

end SSA

/-
inductive SimpleOp : List data → data → Type where
  | iota (n : Nat) : SimpleOp [] ⟨.float, [n]⟩
  | add {σ : data} : SimpleOp [σ, σ] σ

instance (args : List data) (out : data) : ToString (SimpleOp args out) where
  toString x :=
    match x with
    | .iota n => s!"iota {n}"
    | .add => s!"add"

def SimpleOp.expr_add {args : List data} {σ : data} (x y : Expr SimpleOp args [σ]) :
    Expr SimpleOp args [σ] :=
  let feedin : Expr SimpleOp args [σ, σ] := Expr.append x y
  Expr.bind SimpleOp.add feedin

def SimpleOp.expr_iota {args : List data} (n : Nat) :
    Expr SimpleOp args [⟨.float, [n]⟩] :=
  Expr.bind (SimpleOp.iota n) Expr.nil

def foobar (n : Nat) :=
  define_expr using SimpleOp with
    x : ⟨.float, [n]⟩
  begin
    let_expr y : [⟨.float, [n]⟩] := SimpleOp.expr_add x (SimpleOp.expr_iota n);
    return (SimpleOp.expr_add y x).append y


def barfoo (n : Nat) :=
  define_expr using SimpleOp with
    x : ⟨.float, [n]⟩
  begin
    let y := (foobar n).apply x;
    return y

#print foobar

#eval IO.println (barfoo 10).code

end Jax
-/
