import JaxProof.Expr
import JaxProof.Eval

namespace Jax

def curryType (α : Type) : Nat → Type
  | 0 => α
  | n + 1 => α → curryType α n

class Impl (α : ℕ → Type) where
  protected mul {n : ℕ} : α n → α n → α n
  protected cast {n m : ℕ} : α n → n = m → α m
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
  mul x y := ⟨x.expr.mul y.expr⟩
  lift m x := ⟨Nat.repeat Expr.lift m x.expr⟩
  cast x _ := ⟨x.expr⟩
  fori_loop n x f := ⟨Expr.fori_loop n.expr x.expr (.func 2 (f ⟨Expr.arg 0⟩ ⟨Expr.arg 1⟩).expr)⟩

def trace {n : ℕ} (f : {α : ℕ → Type} → [Impl α] → curryType (α n) n) : String :=
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
  cast x _ := x
  fori_loop n x f := match n with
    | .int [n] => Nat.rec x (fun i a ↦ f (.int [i]) a) n.natAbs
    | _ => .error

end Jax

open Jax.Impl

variable {α : ℕ → Type} [Jax.Impl α]

/-

I want to define a DSL supporting following syntax

```
jax_def f(n, x):
  jax_def g(i, a):
    return a * x
  return fori_loop n x g
```

which would be expanded to the following code

-/

def f : Jax.curryType (α 2) 2 := fun n x ↦
  let _n_arg : ℕ := 2 -- or maybe we don't need this
  let g : Jax.curryType (α (_n_arg + 2)) 2 := fun i a ↦
    @id (α (_n_arg + 2)) (a * x)
  @id (α 2) (fori_loop n x g)

#eval IO.print (Jax.trace f)

