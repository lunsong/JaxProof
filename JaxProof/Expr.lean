import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Tuple.Basic

/-!
This file contains the inductive type `JAX.Expr`, which represents a jaxpr, and `JAX.Expr.code`
which generates the python code.
-/

namespace JAX

/--
The representation of JAX expressions. `Expr n` is an expression with n variables.
Each constructor's first argument is the name of the expression, which is used during code
generation
-/
inductive Expr : ℕ → Type where
  /-- `arg i` is the i'th argument -/
  | arg {n : ℕ} : String → Fin n → Expr n
  /-- constant float array -/
  | const_float {n : ℕ} : String → List ℚ → Expr n
  /-- constant integer array -/
  | const_int {n : ℕ} : String → List ℤ → Expr n
  /-- indexing an array, a = b[i] -/
  | idx {n : ℕ} : String → Expr n → Expr n → Expr n
  /-- set items of an array, a = b.set[i].set(c) -/
  | setIdx {n : ℕ} : String → Expr n → Expr n → Expr n → Expr n
  | add {n : ℕ} : String → Expr n → Expr n → Expr n
  | sub {n : ℕ} : String → Expr n → Expr n → Expr n
  | mul {n : ℕ} : String → Expr n → Expr n → Expr n
  | div {n : ℕ} : String → Expr n → Expr n → Expr n
  /-- repeat an array, `jax.numpy.repeat` -/
  | rep {n : ℕ} : String → ℕ → Expr n → Expr n
  /-- jax.lax.fori_loop -/
  | fori_loop {n : ℕ} : String → ℕ → Expr n → Expr (n + 2) → Expr n
deriving DecidableEq

def Expr.name {n : ℕ} : Expr n → String
  | arg name _ => name
  | const_int name _ => name
  | const_float name _ => name
  | idx name _ _ => name
  | setIdx name _ _ _ => name
  | add name _ _ => name
  | sub name _ _ => name
  | mul name _ _ => name
  | div name _ _ => name
  | rep name _ _ => name
  | fori_loop name _ _ _ => name

def Expr.succ {n : ℕ} : Expr n → Expr (n + 1)
  | arg name i => arg name i.succ
  | const_int name x => const_int name x
  | const_float name x => const_float name x
  | idx name x i => idx name x.succ i.succ
  | setIdx name x i y => setIdx name x.succ i.succ y.succ
  | add name x y => add name x.succ y.succ
  | sub name x y => sub name x.succ y.succ
  | mul name x y => mul name x.succ y.succ
  | div name x y => div name x.succ y.succ
  | rep name n x => rep name n x.succ
  | fori_loop name n x f => fori_loop name n x.succ f.succ

theorem Expr.succ.inj {n : ℕ} : Function.Injective (@Expr.succ n) := fun x y h => by
  induction x with
  | arg _ i =>
    cases y
    any_goals cases h
    unfold succ at h
    injection h with h₁ h₂ h₃
    rw [Fin.succ_inj] at h₃
    rw [h₂, h₃]
  | const_int name x
  | const_float name x =>
    cases y
    any_goals cases h
    rfl
  | idx name x i hx hi => 
    cases y
    any_goals cases h
    unfold succ at h
    injection h with h₁ h₂ h₃ h₄
    rw [hx _ h₃, hi _ h₄, h₂]
  | setIdx name x i y hx hi hy =>
    cases y
    any_goals cases h
    unfold succ at h
    injection h with h₁ h₂ h₃ h₄ h₅
    rw [h₂, hx _ h₃, hi _ h₄, hy _ h₅]
  | add name x y hx hy
  | sub name x y hx hy 
  | mul name x y hx hy
  | div name x y hx hy =>
    cases y
    any_goals cases h
    unfold succ at h
    injection h with h₁ h₂ h₃ h₄
    rw [h₂, hx _ h₃, hy _ h₄]
  | rep name n x hx =>
    cases y
    any_goals cases h
    unfold succ at h
    injection h with h₁ h₂ h₃ h₄
    rw [h₂, h₃, hx _ h₄]
  | fori_loop name n x f hx hf =>
    cases y
    any_goals cases h
    unfold succ at h
    injection h with h₁ h₂ h₃ h₄ h₅
    rw [h₂, h₃, hx _ h₄, hf _ h₅]


def Expr.succ_emb {n : ℕ} : Expr n ↪ Expr (n + 1) where
  toFun := Expr.succ
  inj' := Expr.succ.inj

structure CodeGenCtx (n : ℕ) where
  args : Fin n → String
  vars : String → Option ((m : ℕ) × Expr m)
  code : List String

def Expr.insertLine {n m : ℕ} (expr : Expr m) (line : String) :
    EStateM String (CodeGenCtx n) Unit := do
  let ⟨args, vars, code⟩ ← get
  let new_vars := Function.update vars expr.name (.some ⟨m, expr⟩)
  set (CodeGenCtx.mk args new_vars (line :: code))

def writeLine {n : ℕ} (s : String) : EStateM String (CodeGenCtx n) Unit := do
  let ⟨args, vars, code⟩ ← get
  set (CodeGenCtx.mk args vars (s :: code))

def Expr.genCode {n : ℕ} (expr : Expr n) : EStateM String (CodeGenCtx n) Unit := do
  if expr.name.startsWith "__" then throw s!"invalid Expr name {expr.name}"
  let ⟨args, vars, code⟩ ← get
  match vars expr.name with
  | .some expr' =>
    if ⟨n, expr⟩ ≠ expr' then throw s!"different variable use same name {expr.name}"
  | .none =>
    match expr with
    | arg name i =>
      let new_vars := Function.update vars expr.name (.some ⟨n, expr⟩)
      set (CodeGenCtx.mk (Function.update args i name) new_vars code)
    | const_float name x => insertLine expr s!"{name} = array({x}, dtype=float)"
    | const_int name x => insertLine expr s!"{name} = array({x})"
    | idx name x i =>
      x.genCode
      i.genCode
      insertLine expr s!"{name} = {x.name}[{i.name}]"
    | setIdx name x i y =>
      x.genCode
      i.genCode
      y.genCode
      insertLine expr s!"{name} = {x.name}.at[{i.name}].set({y.name})"
    | add name x y =>
      x.genCode
      y.genCode
      insertLine expr s!"{name} = {x.name} + {y.name}"
    | sub name x y =>
      x.genCode
      y.genCode
      insertLine expr s!"{name} = {x.name} - {y.name}"
    | mul name x y =>
      x.genCode
      y.genCode
      insertLine expr s!"{name} = {x.name} * {y.name}"
    | div name x y =>
      x.genCode
      y.genCode
      insertLine expr s!"{name} = {x.name} / {y.name}"
    | rep name n x =>
      x.genCode
      insertLine expr s!"{name} = {x.name}.repeat({n})"
    | fori_loop name m init f =>
      init.genCode
      match vars f.name with
      | .some x =>
        if x ≠ ⟨n + 2, f⟩ then throw s!"different variables have same name {f.name}"
      | .none =>
        let vars' (name : String) : Option ((l : ℕ) × Expr l) :=
          match vars name with
          | .none => .none
          | .some ⟨l, e⟩ => .some ⟨l + 2, e.succ.succ⟩
        let args' : Fin (n + 2) → String :=
          Fin.cons "__unused_i" (Fin.cons "__unused_carrier" args : Fin (n + 1) → String)
        match f.genCode.run ⟨args', vars', []⟩ with
        | .ok _ ⟨args'', _ , code'⟩ =>
          insertLine f s!"def _fun_{f.name}({args'' 0}, {args'' 1}):"
          let code'' := code'.map ("  " ++ ·)
          modify (fun ⟨args, vars, code⟩ =>
            ⟨args, vars, s!"  return {f.name}" :: code'' ++ code⟩)
        | .error msg _ => throw msg
      insertLine expr s!"{name} = fori_loop(0, {m}, _fun_{f.name}, {init.name})"

def Expr.code {n : ℕ} (expr : Expr n) (name : String) : String :=
  let result := expr.genCode.run ⟨fun i ↦ s!"__unused_{i}", fun _ => .none, []⟩
  match result with
  | .ok _ ⟨args, _, code⟩ =>
    let body : String := "\n  ".intercalate code.reverse
    let argnames_list := ", ".intercalate (List.ofFn args)
    s!"def {name}({argnames_list}):\n  {body}\n  return {expr.name}"
  | .error err_msg _ => s!"# error: {err_msg}"

end JAX
