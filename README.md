# JAX-Lean: Verified JAX Code Generation

This project generates JAX Python code from Lean 4 expressions, enabling formal verification of numerical computations.

## Purpose

Write mathematical expressions in Lean 4, prove properties about them, and generate executable JAX code. The `Expr n` type represents JAX computations with `n` arguments, tracking type safety at compile time.

### Example: Repeated Squaring

**Lean definition:**

```Lean
import JaxProof.Api

open Jax.Impl

-- define the jax function
jax_def f(n, x):
  def g(i, a):
    return a * a
  return fori_loop n x g

-- extract python code
#eval IO.println (Jax.trace f).code

-- prove mathematical properties
example (n : ℕ) (x : List ℝ) :
    (Jax.native f) (.int [n]) (.float x) = .float (x.map (· ^ (2 ^ n))) := by
  simp[f]
  induction n with
  | zero => simp
  | succ n ih =>
    simp [ih]
    simp [HMul.hMul, Jax.Array.pairwise, Jax.Array.mul]
    congr
    apply List.ext_get
    · simp
    simp
    intro i _ _
    conv_lhs =>
      change (x[i] ^ 2 ^ n) * (x[i] ^ 2 ^ n)
    rw [pow_add, pow_one, pow_mul, pow_two]
```

Generated JAX code:

```
def x0(x1, x2):
  def x3(_x0, x4):
    x5 = x4 * x4
    return x5
  x4 = jax.lax.fori_loop(0, x1[0], x3, x2)
  return x4
```
