import JaxProof.Expr
import JaxProof.Eval

namespace Jax

class Impl (α : ℕ → Type) where
  lift {n : ℕ} (m : ℕ) : α n → α (n + m)
  protected cast {n m : ℕ} : α n → n = m → α m
  protected add {n : ℕ} : α n → α n → α n
  protected sub {n : ℕ} : α n → α n → α n
  protected mul {n : ℕ} : α n → α n → α n
  protected div {n : ℕ} : α n → α n → α n
  protected mod {n : ℕ} : α n → α n → α n
  protected divInt {n : ℕ} : α n → α n → α n
  protected idx {n : ℕ} : α n → α n → α n
  protected setIdx {n : ℕ} : α n → α n  → α n → α n
  protected select {n : ℕ} : α n → α n → α n → α n
  protected eq {n : ℕ} : α n → α n → α n
  protected lt {n : ℕ} : α n → α n → α n
  rep {n : ℕ} : ℕ → α n → α n
  ofRat {n : ℕ} : List ℚ → α n
  ofInt {n : ℕ} : List ℤ → α n
  iota {n : ℕ} : ℕ → α n
  fori_loop {n : ℕ} : α n → α n → (α (n + 2) → α (n + 2) → α (n + 2)) → α n
  einsum {n : ℕ} (s : List ℕ) : List (List (Fin s.length)) → List (Fin s.length) → List (α n) → α n

@[simp]
def withLift₂ (α : ℕ → Type) [Impl α] (f : {n : ℕ} → α n → α n → α n)
  {n m : ℕ} (x : α n) (y : α m) : α (max n m) :=
  let x' : α (max n m) := Impl.cast (Impl.lift (m - n) x) add_tsub_eq_max
  let y' : α (max n m) := Impl.cast (Impl.lift (n - m) y) (max_comm n m ▸ add_tsub_eq_max)
  f x' y'

@[simp]
def withLift₃ (α : ℕ → Type) [Impl α] (f : {n : ℕ} → α n → α n → α n → α n)
  {n m l : ℕ} (x : α n) (y : α m) (z : α l) : α (max (max n m) l) :=
  let x' : α (max n m) := Impl.cast (Impl.lift (m - n) x) add_tsub_eq_max
  let x''  : α (max (max n m) l) := Impl.cast (Impl.lift (l - max n m) x') add_tsub_eq_max
  let y' : α (max n m) := Impl.cast (Impl.lift (n - m) y) (max_comm n m ▸ add_tsub_eq_max)
  let y''  : α (max (max n m) l) := Impl.cast (Impl.lift (l - max n m) y') add_tsub_eq_max
  let z' : α (max (max n m) l) := Impl.cast (Impl.lift (max n m - l) z) <| by
    rw[max_comm _ l]
    exact add_tsub_eq_max
  f x'' y'' z'

section

variable {α : ℕ → Type} [Impl α] {n m l : ℕ}

def select : α n → α m → α l → α (max (max n m) l) := withLift₃ α Impl.select

instance : HAdd (α n) (α m) (α (max n m)) where
  hAdd := withLift₂ α Impl.add

instance : HSub (α n) (α m) (α (max n m)) where
  hSub := withLift₂ α Impl.sub

instance : HMul (α n) (α m) (α (max n m)) where
  hMul := withLift₂ α Impl.mul

instance : HDiv (α n) (α m) (α (max n m)) where
  hDiv := withLift₂ α Impl.div

def divInt {α : ℕ → Type} [Impl α] {n m : ℕ} : α n → α m → α (max n m) := withLift₂ α Impl.divInt

infix:50 "//" => divInt

instance : HMod (α n) (α m) (α (max n m)) where
  hMod := withLift₂ α Impl.mod

instance : GetElem (α n) (α m) (α (max n m)) (fun _  _ ↦ True) where
  getElem x i _ := withLift₂ α Impl.idx x i

def setIdx : α n → α m → α l → α (max (max n m) l) := withLift₃ α Impl.setIdx

notation:50 a:50 ".at[" i:50 "].set(" b:50 ")" => setIdx a i b

end

structure Tracer (n : ℕ) where
  expr : Expr

instance ImplTracer : Impl Tracer where
  mul x y := ⟨.binop .mul x.expr y.expr⟩
  add x y := ⟨.binop .add x.expr y.expr⟩
  sub x y := ⟨.binop .sub x.expr y.expr⟩
  div x y := ⟨.binop .div x.expr y.expr⟩
  mod x y := ⟨.binop .mod x.expr y.expr⟩
  divInt x y := ⟨.binop .divInt x.expr y.expr⟩
  idx x y := ⟨.binop .idx x.expr y.expr⟩
  setIdx x i y := ⟨.triop .setIdx x.expr i.expr y.expr⟩
  lift m x := ⟨x.expr.lift m⟩
  cast x _ := ⟨x.expr⟩
  ofInt x := ⟨.nullop (.const_int x)⟩
  ofRat x := ⟨.nullop (.const_float x)⟩
  fori_loop n x f := ⟨.triop .fori_loop n.expr x.expr (.fn 2 (f ⟨Expr.arg 0⟩ ⟨Expr.arg 1⟩).expr)⟩
  einsum s i o x := ⟨.varop (.einsum s i o) (x.map Tracer.expr)⟩
  iota n := ⟨.nullop (.iota n)⟩
  rep n x := ⟨.unop (.rep n) x.expr⟩
  select c x y := ⟨.triop .select c.expr x.expr y.expr⟩
  eq x y := ⟨.binop .eq x.expr y.expr⟩
  lt x y := ⟨.binop .lt x.expr y.expr⟩

def trace {n : ℕ} (f : {α : ℕ → Type} → [Impl α] → curryType (α 0) n) : Expr :=
  let α := Tracer 0
  let rec feed {m : ℕ} (f : curryType α m) : α :=
    match m with
    | 0 => f
    | m + 1 => feed (f ⟨.arg (n - m - 1)⟩)
  .fn n (feed f).expr

noncomputable instance ImplArray : Impl (fun _ ↦ Array) where
  mul := Array.mul
  add := Array.add
  sub := Array.sub
  div := Array.div
  mod := Array.mod
  divInt := Array.divInt
  idx := Array.idx
  setIdx := Array.setIdx
  einsum := Array.einsum
  ofInt := Array.int
  ofRat := Array.float ∘ (List.map Rat.cast)
  lift _ := id
  cast x _ := x
  iota n := Array.int <| List.ofFn fun (i : Fin n) ↦ i
  rep := Array.rep
  fori_loop n x f := match n with
    | .int [n] => Nat.rec x (fun i a ↦ f (.int [i]) a) n.natAbs
    | _ => .error
  select := Array.select
  eq := Array.eq
  lt := Array.lt


@[simp]
noncomputable def native {n : ℕ} (f : {α : ℕ → Type} → [Impl α] → curryType (α 0) n) :
  curryType Array n := @f (fun _ ↦ Array) ImplArray


attribute [simp] Impl.fori_loop Impl.ofRat Impl.cast Impl.mul Impl.lift

declare_syntax_cat jax_term

syntax "jax_def" ("(" ident ":" term")")* ident "(" ident,* "):" ppLine jax_term : command
syntax "return" term : jax_term
syntax ident "="  term ";" ppLine jax_term : jax_term
syntax "def" ident "(" ident,* "):" ppLine jax_term ppLine jax_term : jax_term

open Lean in macro_rules
  | `(jax_def $[($spec_n : $spec_t)]* $name ($args,*): $body) => do
    let narg := (args.elemsAndSeps.size + 1) / 2
    let rec parse (narg : ℕ) : TSyntax `jax_term → MacroM (TSyntax `term)
    | `(jax_term|return $t:term) => `(@id (α _) $t)
    | `(jax_term|$assign:ident = $value:term ; $t:jax_term) => do
      let parsed ← parse narg t
      `(let $assign : α $(quote narg) := $value
        $parsed)
    | `(jax_term|def $name:ident ( $args:ident,* ): $value:jax_term $t:jax_term) => do
      let new_arg : ℕ := (args.elemsAndSeps.size + 1) / 2
      let narg' := narg + new_arg
      let content ← parse narg' value
      let parsed ←  parse narg t
      `(let $name : curryType (α $(quote narg')) $(quote new_arg) := fun $args* => $content;
        $parsed)
    | _ => Macro.throwUnsupported
    let parsed ← parse 0 body
    `(def $name $[($spec_n : $spec_t)]* {α : ℕ → Type} [Impl α] :
      curryType (α 0) $(quote narg) := fun $args* => $parsed)
end Jax


