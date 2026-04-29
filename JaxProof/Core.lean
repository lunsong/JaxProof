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
  | apply {args ins outs : List data} : Expr op ins outs → Expr op args ins → Expr op args outs
  | bind {args ins outs : List data} {exprs : List (List data × List data)} :
    op exprs ins outs →
      (∀ i : Fin exprs.length, Expr op exprs[i].1 exprs[i].2) → Expr op args ins → Expr op args outs

section

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
      let exprs := ",".intercalate (exprs.map fun i => s!"&{i}")
      let ins ← ins.genCode
      let out_ids ← expr.addVars s!"{op} {exprs},{",".intercalate ins}"
      return out_ids.map fun n ↦ s!"%{n}"
    | apply f x =>
      let f ← f.addLib
      let x ← x.genCode
      let out_ids ← expr.addVars s!"call &{f},{",".intercalate x}"
      return out_ids.map fun n ↦ s!"%{n}"
      
  | some ⟨_, out_ids, _⟩ =>
    return out_ids.map fun n => s!"%{n}"

end

unsafe def Expr.code {args outs : List data} (expr : Expr op args outs) : String :=
  let ⟨out_names, _, codes, libs⟩ := expr.genCode ⟨0, [], []⟩
  let body := processCode out_names codes
  let libs := "\n\n".intercalate <| List.ofFn fun (i : Fin libs.length) => s!"&{i}\n{libs[i].2}"
  s!"{body}\n\n{libs}"

end

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

def evalType {data : Type} (impl : data → Type) (args outs : List data) : Type :=
  match args with
  | [] => ∀ i : Fin outs.length, impl outs[i]
  | arg :: args => impl arg → evalType impl args outs

def Impl.bindType {data : Type} (impl : data → Type)
  (exprs : List (List data × List data)) (args outs : List data) :=
  match exprs with
  | [] => evalType impl args outs
  | expr :: exprs => evalType impl expr.1 expr.2 → bindType impl exprs args outs

class Impl {data : Type} (op : OpType data) (impl : data → Type) where
  bind {expr : List (List data × List data)} {args outs : List data} : 
    op expr args outs → Impl.bindType impl expr args outs

def evalType.const {data : Type} {impl : data → Type} {args outs : List data}
  (x : ∀ i : Fin outs.length, impl outs[i]) : evalType impl args outs :=
  match args with
  | [] => x
  | _ :: _ => fun _ => const x

def evalType.arg {data : Type} {impl : data → Type} {args : List data} (i : Fin args.length) :
    evalType impl args [args[i]] :=
  match args with
  | _ :: _ =>
    match i with
    | .mk 0 h => fun x => const fun r => match r with | .mk 0 _ => x
    | .mk (i + 1) hi => fun _ => arg <| .mk i <| by simpa using hi

def evalType.append₀ {data : Type} {impl : data → Type} {outs outs' : List data}
    (x : ∀ i : Fin outs.length, impl outs[i]) (y : ∀ i : Fin outs'.length, impl outs'[i])
    (i : Fin (outs ++ outs').length) : impl (outs ++ outs')[i] := 
  match outs with
  | [] => y i
  | out :: outs =>
    match i with
    | .mk 0 h => x <| .mk 0 <| by simp
    | .mk (i + 1) h => append₀ (fun i => x i.succ) y <| .mk i <| by simpa using h

def evalType.append {data : Type} {impl : data → Type} {args outs outs' : List data} :
    evalType impl args outs → evalType impl args outs' → evalType impl args (outs ++ outs') :=
  match args with
  | [] => append₀
  | _ :: _ => fun x y a => append (x a) (y a)

def evalType.select₀ {data : Type} {impl : data → Type} {outs : List data} (i : List (Fin outs.length))
  (x : ∀ i : Fin outs.length, impl outs[i]) (r : Fin (i.map outs.get).length) : impl (i.map outs.get)[r] :=
  match i with
  | i₀ :: i =>
    match r with
    | .mk 0 h => x i₀
    | .mk (r + 1) h => select₀ i x <| .mk r <| by simpa using h

def evalType.select {data : Type} {impl : data → Type} {args outs : List data} (i : List (Fin outs.length)) :
    evalType impl args outs → evalType impl args (i.map outs.get) :=
  match args with
  | [] => select₀ i
  | _ :: _ => fun x a => select i (x a)

def evalType.bind {data : Type} {impl : data → Type} {exprs : List (List data × List data)} {args outs : List data} :
    Impl.bindType impl exprs args outs → (∀ i : Fin exprs.length, evalType impl exprs[i].1 exprs[i].2) → evalType impl args outs :=
  match exprs with
  | [] => fun x _ => x
  | expr :: exprs => fun op fs => bind (op (fs ⟨0, by simp⟩)) (fun i => fs i.succ)

def evalType.apply₀ {data : Type} {impl : data → Type} {ins outs : List data} :
    evalType impl ins outs → (∀ i : Fin ins.length, impl ins[i]) → (∀ i : Fin outs.length, impl outs[i]) :=
  match ins with
  | [] => fun x _ => x
  | in₀ :: ins => fun f x => apply₀ (f (x ⟨0, by simp⟩)) fun i => x i.succ

def evalType.apply {data : Type} {impl : data → Type} {args ins outs : List data} :
    evalType impl ins outs → evalType impl args ins → evalType impl args outs :=
  match args with
  | [] => apply₀
  | _ :: _ => fun f x a => apply f (x a)

def Expr.eval {data : Type} {opType : OpType data} {args outs : List data}
  (impl : data → Type) [Impl opType impl] : Expr opType args outs → evalType impl args outs
  | arg i => evalType.arg i
  | append x y =>  evalType.append (x.eval impl) (y.eval impl)
  | select i x => evalType.select i (x.eval impl)
  | bind op fs xs =>
    let op := Impl.bind (impl := impl) op
    let op := evalType.bind op fun i => (fs i).eval impl
    evalType.apply op (xs.eval impl)
  | apply f xs => evalType.apply (f.eval impl) (xs.eval impl)

inductive SimpleOp (data : Type) : List (List data × List data) → List data → List data → Type
  | node {α : data} : SimpleOp data [] [α, α] [α]

instance (exprs : List (List String × List String)) (args outs : List String) :
    ToString (SimpleOp String exprs args outs) where
  toString _ := "node"

def foobar :=
  ssa SimpleOp String with x : "Float", y : "Float" begin
  let_expr z : ["Float"] := .bind .node (fun i => nomatch i) (.append x y);
  return z

def barfoo :=
  ssa SimpleOp String with x : "Float", y : "Float" begin
  let_expr z : ["Float"] := foobar.apply (x.append y);
  return z

#eval IO.println barfoo.code

end SSA


