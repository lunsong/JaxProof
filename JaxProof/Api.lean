import JaxProof.Expr
import JaxProof.Eval

namespace Jax

class Impl (α : ℕ → Type) where
  protected mul {n : ℕ} : α n → α n → α n
  protected cast {n m : ℕ} : α n → n = m → α m
  ofRat {n : ℕ} : List ℚ → α n
  lift {n : ℕ} (m : ℕ) : α n → α (n + m)
  fori_loop {n : ℕ} : α n → α n → (α (n + 2) → α (n + 2) → α (n + 2)) → α n

instance (α : ℕ → Type) [Impl α] (n m : ℕ) : HMul (α n) (α m) (α (max n m)) where
  hMul x y := 
    if h : n ≤ m then
      let x' := Impl.lift (m - n) x
      let y' := Impl.cast y (Nat.add_sub_of_le h).symm
      Impl.cast (Impl.mul x' y') add_tsub_eq_max
    else
      let x' := Impl.cast x (Nat.add_sub_of_le (lt_of_not_ge h).le).symm
      let y' := Impl.lift (n - m) y
      Impl.cast (Impl.mul x' y') (max_comm n m ▸ add_tsub_eq_max)

structure Tracer (n : ℕ) where
  expr : Expr

instance ImplTracer : Impl Tracer where
  mul x y := ⟨.binop .mul x.expr y.expr⟩
  lift m x := ⟨x.expr.lift m⟩
  cast x _ := ⟨x.expr⟩
  ofRat x := ⟨.nullop (.const_float x)⟩
  fori_loop n x f := ⟨.triop .fori_loop n.expr x.expr (.fn 2 (f ⟨Expr.arg 0⟩ ⟨Expr.arg 1⟩).expr)⟩

def trace {n : ℕ} (f : {α : ℕ → Type} → [Impl α] → curryType (α n) n) : Expr :=
  let α := Tracer n
  let rec feed {m : ℕ} (f : curryType α m) : α :=
    match m with
    | 0 => f
    | m + 1 => feed (f ⟨.arg (n - m - 1)⟩)
  .fn n (feed f).expr

instance ImplArray : Impl (fun _ ↦ Array) where
  mul := Array.mul
  ofRat x := .float (x.map Rat.cast)
  lift _ := id
  cast x _ := x
  fori_loop n x f := match n with
    | .int [n] => Nat.rec x (fun i a ↦ f (.int [i]) a) n.natAbs
    | _ => .error

@[simp]
def native {n : ℕ} (f : {α : ℕ → Type} → [Impl α] → curryType (α n) n) : curryType Array n :=
  @f (fun _ ↦ Array) ImplArray


attribute [simp] Impl.fori_loop Impl.ofRat Impl.cast Impl.mul Impl.lift

/-
macro "jax_def" name:ident ":" body:term : command => do
  `(def $name : ℕ := let c : ℕ := 0; $body)

jax_def f:
  c
-/

declare_syntax_cat jax_term

syntax "jax_def" ident "(" ident,* "):" ppLine jax_term : command
syntax "return" term : jax_term
syntax ident "=" term ppLine jax_term : jax_term
syntax "def" ident "(" ident,* "):" ppLine jax_term ppLine jax_term : jax_term

open Lean in macro_rules
  | `(jax_def $name ($args,*): $body) => do
    let narg := (args.elemsAndSeps.size + 1) / 2
    let rec parse (narg : ℕ) : TSyntax `jax_term → MacroM (TSyntax `term)
    | `(jax_term|return $t:term) => `(term|@id (α _) $t)
    | `(jax_term|$assign:ident = $value:term $t:jax_term) => do
      let parsed ← parse narg t
      `(term|let $assign : α $(quote narg) := $value; $parsed)
    | `(jax_term|def $name:ident ( $args:ident,* ): $value:jax_term $t:jax_term) => do
      let new_arg : ℕ := (args.elemsAndSeps.size + 1) / 2
      let narg' := narg + new_arg
      let content ← parse narg' value
      let parsed ←  parse narg t
      `(let $name : curryType (α $(quote narg')) $(quote new_arg) := fun $args* => $content;
        $parsed)
    | _ => Macro.throwUnsupported
    let parsed ← parse narg body
    `(def $name {α : ℕ → Type} [Impl α] : curryType (α $(quote narg)) $(quote narg) :=
        fun $args* => $parsed)
end Jax

open Jax.Impl

--def f (m : ℕ) {α : ℕ → Type} [Jax.Impl α] : Jax.curryType (α 2) 2 := fun n x ↦
--  let one : α 2 := ofRat <| List.ofFn fun (_ : Fin m) ↦ 1
--  let g (_ a : α 4) : α 4 := (x * x) * a
--  @id (α _) (fori_loop n one g)

jax_def f(n, x):
  def g(i, a):
    return a * a
  return fori_loop n x g



#eval IO.println (Jax.trace f).outward.code

example (n m : ℕ) (x : List ℝ) (h : x.length = m) :
    (Jax.native (f m)) (.int [n]) (.float x) = .float (x.map (· ^ (2 * n))) := by
  simp[f]
  induction n with
  | zero => simp[h]
  | succ n ih =>
    simp [ih]
    simp [HMul.hMul, Jax.Array.pairwise, Jax.Array.mul]
    congr
    apply List.ext_get
    · simp
    simp
    intro i _ _
    change (x[i] * x[i]) * (x[i] ^ (2 * n)) = x[i] ^ (2 * (n + 1))
    rw [mul_add, mul_one, pow_add _ _ 2, mul_comm _ (x[i] ^2), pow_two]


