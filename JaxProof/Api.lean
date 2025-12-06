import JaxProof.Expr
import JaxProof.Eval

namespace Jax

def curryType (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => α → curryType α n

class Impl (α : ℕ → Type) where
  mul {n : ℕ} : α n → α n → α n
  lift {n : ℕ} : α n → α (n + 1)
  fori_loop {n : ℕ} : α n → α n → (α (n + 2) → α (n + 2) → α (n + 2)) → α n

section inst

variable (α : ℕ → Type) [Impl α] (n m : ℕ)

instance : HMul (α n) (α (n + m)) (α (n + m)) where
  hMul x y := Impl.mul (Nat.rec x (fun _ x ↦ Impl.lift x) m) y

instance : HMul (α (n + m)) (α n) (α (n + m)) where
  hMul x y := Impl.mul x (Nat.rec y (fun _ y ↦ Impl.lift y) m)

end inst

structure Tracer (n : ℕ) where
  expr : Expr

instance : Impl Tracer where
  mul x y := ⟨x.expr.mul y.expr⟩
  lift x := ⟨x.expr.lift⟩
  fori_loop n x f := ⟨Expr.fori_loop n.expr x.expr (f ⟨Expr.arg 0⟩ ⟨Expr.arg 1⟩).expr⟩

def f {α : ℕ → Type} [Impl α] (x n : α 2) : α 2 :=
  let g (i a : α 4) : α 4 := x * a
  Impl.fori_loop n x g

end Jax
