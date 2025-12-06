import JaxProof.Expr
import JaxProof.Eval

namespace Jax

def curryType (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => α → curryType α n

class Impl (α : ℕ → Type) where
  protected mul {n : ℕ} : α n → α n → α n
  lift {n : ℕ} (m : ℕ) : α n → α (n + m)
  fori_loop {n : ℕ} : α n → α n → (α (n + 2) → α (n + 2) → α (n + 2)) → α n

instance (α : ℕ → Type) [Impl α] (n : ℕ) : Mul (α n) where mul := Impl.mul

structure Tracer (n : ℕ) where
  expr : Expr

instance ImplTracer : Impl Tracer where
  mul x y := ⟨x.expr.mul y.expr⟩
  lift m x := ⟨Nat.repeat Expr.lift m x.expr⟩
  fori_loop n x f := ⟨Expr.fori_loop n.expr x.expr (.func 2 (f ⟨Expr.arg 0⟩ ⟨Expr.arg 1⟩).expr)⟩

def trace (n : ℕ) (f : {α : ℕ → Type} → [Impl α] → curryType (α n) n) : String :=
  let α := Tracer n
  let rec feed {m : ℕ} (f : curryType α m) : α :=
    match m with
    | 0 => f
    | m + 1 => feed (f ⟨.arg (n - m - 1)⟩)
  let expr := Expr.func n (feed f).expr
  expr.code

instance ImplArray : Impl (fun _ ↦ Array) where
  mul := Array.mul
  lift _ := id
  fori_loop n x f := match n with
    | .int [n] => Nat.rec x (fun i a ↦ f (.int [i]) a) n.natAbs
    | _ => .error

end Jax

open Jax.Impl

variable {α : ℕ → Type} [Jax.Impl α]

def f (n x : α 2) : α 2 :=
  let g (_ a : α 4) : α 4 := a * lift 2 x
  fori_loop n x g

#eval IO.println (Jax.trace 2 f)

