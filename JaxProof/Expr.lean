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
  | arg {args : List data} (i : Fin args.length) : Expr op ho args [args[i]]
  | call {args ins outs: List data} :
    Expr op ho ins outs → Expr op ho args ins  → Expr op ho args outs
  | append {args outs outs' : List data} :
    Expr op ho args outs → Expr op ho args outs' → Expr op ho args (outs ++ outs')
  | select {args outs : List data} (i : List (Fin outs.length)) :
    Expr op ho args outs → Expr op ho args (i.map outs.get)
  | bindOp {args ins outs: List data} :
    op ins outs → Expr op ho args ins → Expr op ho args outs
  | bindHo {args outs : List data} {ins : List (List data × List data)} :
    ho ins args outs → (∀ i : Fin ins.length, Expr op ho ins[i].1 ins[i].2) → Expr op ho args outs

variable {data : Type}
variable {op : List data → List data → Type} [∀ args, ∀ out, ToString (op args out)]
variable {ho : List (List data × List data) → List data → List data → Type}

abbrev Cached (α : Type) : Type := List (USize × α)

abbrev Expr.CodeM (α : Type) : Type :=
  StateM (Nat × Cached (List Nat × String) × Cached String) α

unsafe def Expr.addLine {args outs : List data}
  (expr : Expr op ho args outs) (code : String) : CodeM (List Nat) :=
  fun ⟨n_var, codes, libs⟩ ↦
    match codes.find? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let out_ids : List Nat := List.ofFn fun (i : Fin outs.length) ↦ i.val + n_var
      let new_codes := codes.concat ⟨ptrAddrUnsafe expr, out_ids, code⟩
      ⟨out_ids, outs.length + n_var, new_codes, libs⟩
    | some ⟨_, out_ids, _⟩ =>
      ⟨out_ids, n_var, codes, libs⟩

def Expr.processCode (out_names : List String) (codes : Cached (List Nat × String)) : String :=
  let body : String := "\n".intercalate <|
    codes.map fun ⟨_, assign_id, line⟩ =>
      let assign_names := ",".intercalate (assign_id.map fun i => s!"%{i}")
      s!"{assign_names} = {line}"
  s!"{body}\nreturn {",".intercalate out_names}"

mutual

unsafe def Expr.addLib {args outs : List data}
  (expr : Expr op args outs) : CodeM Nat :=
  fun ⟨n_var, codes, libs⟩ ↦
    match libs.findIdx? (fun ⟨addr, _⟩ ↦ ptrAddrUnsafe expr == addr) with
    | none =>
      let ⟨out_names, _, fn_codes, libs⟩ := expr.genCode ⟨0, [], libs⟩
      let fn_code := processCode out_names fn_codes
      ⟨libs.length, n_var, codes, libs.concat ⟨ptrAddrUnsafe expr, fn_code⟩⟩
    | some i =>
      ⟨i, n_var, codes, libs⟩

unsafe def Expr.genCode {args outs : List data} (expr : Expr op args outs) :
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

unsafe def Expr.code {args outs : List data} (expr : Expr op args outs) : String :=
  let ⟨out_names, _, codes, libs⟩ := expr.genCode ⟨0, [], []⟩
  let body := processCode out_names codes
  let libs := "\n\n".intercalate <| List.ofFn fun (i : Fin libs.length) => s!"@{i}\n{libs[i].2}"
  s!"{body}\n\n{libs}"

declare_syntax_cat expr_builder

syntax "define_expr" "using" term "with" ( ident ":" term ),*
       "begin" expr_builder : term

syntax "let_expr" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":" term ":=" term ";" expr_builder : expr_builder
syntax "let" ident ":=" term ";" expr_builder : expr_builder
syntax "return" term : expr_builder

open Lean in
partial def parse_expr_builder (op args : TSyntax `term) :
    TSyntax `expr_builder → MacroM (TSyntax `term)
  | `(expr_builder| let_expr $name : $outs := $val; $content) => do
    `(term| let $name : Expr $op $args $outs := $val;
            $(← parse_expr_builder op args content))
  | `(expr_builder| let $name : $type := $val; $content) => do
    `(term| let $name : $type := $val; $(← parse_expr_builder op args content))
  | `(expr_builder| let $name := $val; $content) => do
    `(term| let $name := $val; $(← parse_expr_builder op args content))
  | `(expr_builder| return $rets) => pure rets
  | _ => Macro.throwUnsupported

open Lean in macro_rules
  | `(define_expr using $op with $[$argnames : $argtypes],* begin $body) => do
    let args : TSyntax `term ← `(term| [ $[$argtypes],* ])
    let parsed_body : TSyntax `term ← parse_expr_builder op args body
    let rec bind_args : List (TSyntax `ident × TSyntax `term) → Nat → MacroM (TSyntax `term)
      | [], _ => return parsed_body
      | ⟨name, out⟩ :: rest, n => do
        `(term| let $name : Expr $op $args [$out] := Expr.arg $(quote n);
        $(← bind_args rest (n + 1)))
    bind_args (argnames.toList.zip argtypes.toList) 0

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
