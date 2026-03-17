# JaxProof: Verified JAX/XLA Code Generation from Lean 4

This is my personal project to generate XLA functions from Lean 4, so that we can write mathematically correct code, and find mistakes before wasting many GPU hours. Any contribution is welcome!

## Example

### 1. MatMul

First we define the matmul using the `xla with` DSL. The `dot_general` is a XLA primitive, it is called with `dot_general batched_dims contracted_dims rest_of_lhs_dims rest_of_rhs_dims x y` where `x` has shape `batched_dims + contracted_dims + rest_of_lhs_dims` and `y` has shape `batched_dims + contracted_dims + rest_of_rhs_dims`. While the original XLA primitive is more flexible, the current implementation in JaxProof requires to transpose the arguments first.

```lean
import JaxProof

def matmul {n m l : ℕ} :=
  xla with
    x : float [n, m],
    y : float [m, l]
  returns
    float [n, l]
  begin
    let_expr x : float [m, n] := Jax.transpose [0,1].formPerm x;
    let_expr z : float [n, l] := Jax.dot_general [] [m] [n] [l] x y;
    return z
```

Next we inspect the IR of this XLA function. 

```lean
#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code
```

The output is

```
transpose [1, 0] $0
dot_general 1 0 %0 $1
return %1, 
with
```

Names like "$1" represents the second argument and "%0" represents the first intermediate variable, i.e. the result of `transpose [1, 0] $0`

We can evaluate the XLA functions on native tensors. Here we prove that this function, when evaluated treating float numbers as real numbers, is equal to the matrix multiplication in Mathlib.

```lean
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval]
  apply funext
  intro i
  apply funext
  intro j
  conv_lhs =>
    change (∑ k, fun i j => x i k * y k j) i j
  simp [Matrix.mul_apply, Finset.sum_apply]
```

### 2. fori_loop

Next we show how to build higher order functions by nesting `xla with` syntax.

```lean
import JaxProof

def fn (m n : ℕ) :=
  xla with
    x : float [n]
  returns
    float [n]
  begin
   let loop_fn :=
     xla with
       i : int [],
       x : float [n]
     returns
       float [n]
     begin
       return .bind .mul *[x, x];
  fori_loop m, loop_fn, (.cons x .nil), .nil

#eval IO.println (fn 2 3).code
```

The generated IR is

```
return fori_loop(2, @0, ($0, ), ())
with
mul $1 $1
return %0, 
```

Note that the content after `with` is the lib of this module, and `@0` represents that fist function in the lib. As above, we can prove some properties about this function.

```
example (m n : ℕ) (x : Fin n → ℝ) :
    ((fn m n).eval Jax.FloatAsReal) *[x] = *[x ^ (2 ^ m)] := by
  simp [Jax.ExprGroup.eval, fn]
  induction m with
  | zero =>
    simp [Jax.Expr.eval]
    rfl
  | succ m ih =>
    simp only [Fin.isValue, ih, Jax.DList.cons.injEq, and_true]
    simp [Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval]
    rw [pow_add, pow_one, pow_mul, pow_two]
    rfl

```

## ToDo

1. The interpreter of the IR is still not implemented. We have three choices: a Python interpreter, a standalone C++ interpreter or using the Lean FFI.

2. Currently both the `Jax.Expr` and `Jax.FloatAsReal` support only a part of XLA primitives. Full support will be implemented in the future.

3. Besides `FloatAsReal`, we can also treat the float numbers as intervals or random variables.
