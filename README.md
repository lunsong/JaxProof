# JAX-Lean: Verified JAX Code Generation

This project generates JAX Python code from Lean 4 expressions, enabling formal verification of numerical computations.

## Purpose

Write mathematical expressions in Lean 4, prove properties about them, and generate executable JAX code. The `Expr n` type represents JAX computations with `n` arguments, tracking type safety at compile time.

### Example: Repeated Squaring

**Lean definition:**

```Lean
import JaxProof

open JAX

namespace test_fun

-- define the JAX Expr
def n : Expr 2 := .arg "n" 0
def x : Expr 2 := .arg "x" 1
def loop_fun_carrier : Expr 4 := .arg "a" 1
def loop_fun : Expr 4 := .mul "b" loop_fun_carrier loop_fun_carrier
def y : Expr 2 := .fori_loop "y" n x loop_fun

end test_fun

-- generate python code
#eval IO.println (test_fun.y.code "f")

-- prove properties about this function
example (n : ℕ) (x : List ℝ) :
    test_fun.y.eval' (.int [n]) (.float x) = .float (x.map (· ^ (2 ^ n))) := by
  simp[test_fun.y, test_fun.n, test_fun.x, test_fun.loop_fun, test_fun.loop_fun_carrier,
    Expr.eval', curry, Expr.eval]
  induction n with
  | zero => simp
  | succ n ih =>
    simp[ih]
    generalize hx' : (List.map (fun x ↦ x ^ 2 ^ n) x) = x'
    generalize hx'' : (List.map (fun x ↦ x ^ 2 ^ (n + 1)) x) = x''
    simp[Array.mul]
    congr
    refine List.ext_get ?_ ?_
    · simp[← hx', ← hx'']
    · intro m h₁ h₂
      simp[← hx', ← hx'']
      conv_rhs =>
        conv =>
          arg 2
          rw [pow_add, pow_one, mul_two]
        rw [pow_add]
```

Generated JAX code:

```
def f(n, x):
  def b(__unused_i, a):
    b = a * a
    return b
  y = fori_loop(0, n[0], b, x)
  return y
```
