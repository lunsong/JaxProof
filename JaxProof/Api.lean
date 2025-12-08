import JaxProof.Expr
import JaxProof.Eval

namespace Jax

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

@[simp]
def native {n : ℕ} (f : {α : ℕ → Type} → [Impl α] → curryType (α n) n) : curryType Array n :=
  @f (fun _ ↦ Array) ImplArray

end Jax

open Jax.Impl

def f {α : ℕ → Type} [Jax.Impl α] : Jax.curryType (α 2) 2 := fun n x ↦
  let g (i a : α 4) : α 4 := (x * x) * a
  @id (α _) (fori_loop n x g)

#eval IO.println (Jax.trace f)

attribute [simp] Jax.Impl.fori_loop

example (n : ℕ) (x : List ℝ) :
    (Jax.native f) (.int [n]) (.float x) = .float (x.map (· ^ (2 * n + 1))) := by
  simp[f]
  induction n with
  | zero => simp
  | succ n ih =>
    simp [ih]
    simp [HMul.hMul, Jax.Impl.cast, Jax.Impl.mul, Jax.Array.pairwise,
      Jax.Impl.lift, Jax.Array.mul]
    congr
    apply List.ext_get
    · simp
    simp
    intro i _ _
    change (x[i] * x[i]) * (x[i] ^ (2 * n + 1)) = x[i] ^ (2 * (n + 1) + 1)
    rw [mul_add, mul_one, add_assoc, add_comm 2, ← add_assoc, pow_add _ _ 2,
      mul_comm _ (x[i] ^2), pow_two]


