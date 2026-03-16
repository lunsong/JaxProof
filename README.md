# JaxProof: Verified JAX/XLA Code Generation from Lean 4

[![Lean Version](https://img.shields.io/badge/Lean-v4.25.0-blue)](https://github.com/leanprover/lean4)
[![Mathlib](https://img.shields.io/badge/Mathlib-v4.25.0-green)](https://github.com/leanprover-community/mathlib4)

This project generates verified JAX/XLA Python code from Lean 4 expressions, enabling formal verification of numerical computations and machine learning operations. Write mathematical expressions in Lean 4, prove properties about them, and generate executable JAX code with correctness guarantees.

## Key Features

- **Two APIs**: High-level `jax_def` DSL (legacy) and XLA-style DSL (new)
- **70+ JAX/XLA Operations**: Including `dot_general`, `einsum`, `broadcast`, `transpose`, `fori_loop`, linear algebra ops
- **Verified Examples**: Softmax, matrix multiplication, vector normalization with correctness proofs
- **Tensor Infrastructure**: N-dimensional tensors with shape tracking, Einstein summation, broadcasting
- **Higher-Order Functions**: Support for loops with carried state via `ExprGroup`

## Project Structure

```
JaxProof/
├── JaxProof/
│   ├── Tensor.lean      # N-dimensional tensor definition and operations
│   ├── Expr.lean        # XLA expression types and code generation
│   ├── Eval.lean        # Expression evaluation semantics
│   ├── Impl.lean        # Native tensor implementation (ℝ and ℤ)
│   ├── Native.lean      # Array operations and conversion
│   ├── Api.lean         # Legacy high-level API with jax_def syntax
│   └── TensorLike.lean  # Tensor abstraction and examples
├── Attention.lean       # Verified softmax implementation
├── example.lean         # Examples: fori_loop, matmul, normalization
└── Tests/
    ├── fori_loop.lean   # Verified repeated squaring
    └── matmul.lean      # Verified matrix multiplication
```

## Quick Start

### XLA-Style DSL (Recommended)

Define computations using the new XLA-style syntax:

```lean
import JaxProof

def matmul {n m l : ℕ} :=
  xla with
    x : float [n, m],
    y : float [m, l]
  returns
    float [n, l]
  begin
    let_expr z : float [n, l] := .bind (.dot_general [] [m] [n] [l]) *[x, y];
    return z

-- Generate Python code
#eval IO.println (matmul.code)

-- Prove correctness
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval, Matrix.mul_apply]
```

### Legacy jax_def DSL

For simpler computations, use the `jax_def` syntax:

```lean
import JaxProof.Api

open Jax.Impl

-- Define a JAX function with fori_loop
jax_def f(n, x):
  def g(i, a):
    return a * a
  return fori_loop n x g

-- Extract Python code
#eval IO.println (Jax.trace f).code

-- Prove: f(n, x) computes x^(2^n)
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

```python
def x0(_x0, x1):
  x2 = x1 * x1
  return x2
def x1(x2, x3):
  x4 = jax.lax.fori_loop(0, x2[0], x0, x3)
  return x4
```

## Verified Examples

### Softmax

```lean
import JaxProof
open Jax.Api

jax_def (n₁ : ℕ) (n₂ : ℕ) softmax(x):
  y = exp x;
  y' = einsum [n₁, n₂] [[#0, #1]] 1 [y];  -- sum over axis
  y' = rep n₁ y';                           -- broadcast
  return y / y'

-- Prove equivalence to mathematical definition
theorem softmax_eq_def (n₁ n₂ : ℕ) (x : Matrix (Fin n₁) (Fin n₂) ℝ) :
    Jax.native (softmax n₁ n₂) (Jax.Array.ofMatrix x) = 
    Jax.Array.ofMatrix (Matrix.of fun i j ↦ Real.exp (x i j) / ∑ k, Real.exp (x k j)) := by
  -- Proof uses simp with Tensor lemmas
  simp [softmax, Jax.Array.einsum, Jax.Tensor.einprod]
```

### Matrix Multiplication

```lean
def matmul {n m l : ℕ} :=
  xla with
    x : float [n, m],
    y : float [m, l]
  returns
    float [n, l]
  begin
    let_expr z : float [n, l] := .bind (.dot_general [] [m] [n] [l]) *[x, y];
    return z

-- Proof of equivalence to Mathlib's matrix multiplication
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval, Jax.Expr.eval, Matrix.mul_apply, Finset.sum_apply]
```

## Supported Operations

### Tensor Operations
- `dot_general` - Generalized contraction (matrix multiplication, batch operations)
- `einsum` - Einstein summation convention
- `broadcast` - Shape broadcasting
- `transpose` - Axis permutation
- `sum` - Summation over leading dimensions
- `concat` - Concatenation along an axis

### Element-wise Operations
- Arithmetic: `add`, `sub`, `mul`, `div`, `mod`, `div_int`
- Math: `exp`, `log`, `sin`, `cos`, `sqrt`, `abs`
- Comparison: `eq`, `lt`, `gt`, `le`, `ge`
- Logical: `select` (conditional)

### Control Flow
- `fori_loop` - Counted loop with carried state

### Linear Algebra
- `cholesky` - Cholesky decomposition
- `eigvals`, `eigvalsh` - Eigenvalues
- `eigvecs`, `eigvecsh` - Eigenvectors

### Other
- `iota` - Range generation
- `argmax`, `argmin` - Index of max/min
- `dynamic_slice`, `dynamic_update_slice`
- Various FFT and convolution operations (WIP)

## Technical Highlights

### Type-Safe Tensors

Tensors are indexed by shape at the type level:

```lean
def Tensor (R : Type) : List ℕ → Type
  | [] => R
  | n₀ :: ns => Fin n₀ → Tensor R ns
```

### Verified Tensor Operations

Key lemmas for tensor manipulation:

```lean
@[simp]
theorem Tensor.flatten_unflatten (s : List ℕ) (x : Fin s.prod → R) :
    (Tensor.unflatten s x).flatten = x

@[simp]
theorem Tensor.unflatten_flatten {s : List ℕ} (x : Tensor R s) :
    Tensor.unflatten s x.flatten = x
```

### Expression Evaluation

Two evaluation backends:
- `FloatAsReal`: Uses Mathlib's `ℝ` and `ℤ` for proofs
- Code generation: Produces JAX Python code

```lean
def Expr.eval {args : List TensorType} {out : TensorType} (impl : TensorType → Type)
  [TensorImpl impl] (xs : DList impl args) (expr : Expr args out) : impl out
```

## Building

Requires Lean 4.25.0 and Mathlib:

```bash
lake update
lake build
```

## Recent Progress

Recent commits have focused on:

1. **New XLA-style DSL**: Complete redesign with `xla with ... begin ... end` syntax
2. **Higher-Order Functions**: `ExprGroup` type supporting `fori_loop` with closures
3. **70+ XLA Operations**: Comprehensive coverage of JAX primitives
4. **Verified Examples**: Softmax and matrix multiplication with full proofs
5. **Computable Evaluation**: All tensor operations are computable for testing
6. **Code Generation**: Working Python code generation for simple expressions

## Limitations and Future Work

- Code generation for higher-order functions (loops with closures) is WIP
- Some linear algebra operations need completion
- Performance optimization for large tensors
- Integration with actual JAX runtime for execution

## License

MIT License - See repository for details.

## Acknowledgments

Built on [Lean 4](https://github.com/leanprover/lean4) and [mathlib4](https://github.com/leanprover-community/mathlib4).
