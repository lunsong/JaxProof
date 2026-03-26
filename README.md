# 🔬 JaxProof

> **Verified XLA Code Generation from Lean 4**

[![Lean](https://img.shields.io/badge/Lean-4-blue)](https://lean-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Write mathematically correct GPU/TPU code. Prove it before running it. Never waste GPU hours on bugs again.

---

## ✨ What is JaxProof?

**JaxProof** is a formal verification framework that bridges the gap between mathematical specifications and high-performance XLA (Accelerated Linear Algebra) code. It allows you to:

- 📝 Write XLA programs in Lean 4 using an elegant DSL
- 🎯 Generate XLA intermediate representation (IR)
- ✅ Formally prove that your code matches mathematical specifications
- 🚀 Run with confidence on JAX/XLA backends

---

## 🚀 Quick Start

### Matrix Multiplication

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

-- Generate XLA IR
#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).pretty_print
```

Output:
```
%0: transpose [1, 0] $0
%1: dot_general 1 0 %0 $1
return %1, 
```

### Prove Correctness

```lean
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval]
  apply funext; intro i
  apply funext; intro j
  conv_lhs => change (∑ k, fun i j => x i k * y k j) i j
  simp [Matrix.mul_apply, Finset.sum_apply]
```

💡 **The proof guarantees** that the generated XLA code computes the exact mathematical matrix product.

---

## 📚 Examples Gallery

### 🔁 Counted Loops with `fori_loop`

Square a vector `m` times:

```lean
def power_loop {n : ℕ} :=
  xla with
    x : float [n],
    m : int []
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
    fori_loop loop_fn, m, (.cons x .nil), .nil
```

**Generated IR:**
```
return fori_loop($1, , @0, ($0, ), ())
@0:
%0: mul $1 $1
returns %0, 
```

**Theorem:** `(power_loop.eval Jax.FloatAsReal) *[x, m] = *[x ^ (2 ^ m)]`

---

### 📐 Vector Normalization

Compute L2 norm and normalize a vector:

```lean
def norm_xla {n : ℕ} :=
  xla with
    x : float [n]
  returns
    float []
  begin
    let_expr x2 : float [n] := .bind .mul *[x, x];
    let_expr x2_sumed : float [] := .bind (.sum 1) *[x2];
    return .bind .sqrt *[x2_sumed]

def normalize_xla {n : ℕ} :=
  let f₀ := Jax.ExprGroup.cons (Jax.Expr.arg 0) (norm_xla (n := n))
  let f₁ :=
    xla with
      x : float [n],
      norm_x : float []
    returns
      float [n]
    begin
      let_expr norm_x : float [n] := .bind (.broadcast [(n, false)]) *[norm_x];
      return .bind .div *[x, norm_x]
  Jax.ExprGroup.apply f₀ f₁
```

**Proof:** `normalize_xla.eval Jax.FloatAsReal *[x] = *[x / ‖x‖₂]`

---

### 🎯 Inverse Permutation with Scatter

```lean
def inv_permutation {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    return .bind .scatter *[0, Jax.iota, x]
```

If `x` represents a permutation `σ`, this computes `σ⁻¹`.

---

### 🔍 Index of Nonzero Elements

```lean
def idxOfNonzero {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    let_expr mask : int [n] := Jax.choice x 1 0;      -- 1 where x≠0, else 0
    let_expr counts : int [n] := Jax.cumsum mask;     -- running count
    let_expr result : int [n] := Jax.choice x counts 0; -- keep counts where x≠0
    return result
```

---

## 🧮 Supported XLA Operations

<details>
<summary><b>⚡ Element-wise Operations</b></summary>

| Unary | Binary |
|-------|--------|
| `abs`, `neg`, `sqrt`, `cbrt` | `add`, `sub`, `mul`, `div` |
| `exp`, `exp2`, `expm1` | `div_int`, `mod` |
| `cos`, `sin`, `acos`, `asin` | `and`, `eq` |
| `cosh`, `sinh`, `acosh`, `asinh` | |
| `atan`, `atanh` | |
| `erf`, `erfc`, `erf_inv` | |
| `bessel_i0e`, `bessel_i1e` | |
| `ceil` | |

</details>

<details>
<summary><b>📊 Reduction & Scan</b></summary>

- `sum n` — sum over leading `n` dimensions
- `argmax axis`, `argmin axis` — index of max/min
- `cumsum`, `cummax`, `cummin`, `cumprod`, `cumlogsumexp`

</details>

<details>
<summary><b>🔄 Shape Manipulation</b></summary>

- `transpose perm` — arbitrary dimension permutation
- `broadcast dims` — expand dimensions with broadcasting
- `concat axis` — concatenate along any axis
- `dynamic_slice`, `dynamic_update_slice`

</details>

<details>
<summary><b>🔢 Linear Algebra</b></summary>

- `dot_general batch contract lhs rhs` — generalized dot product
- `cholesky` — Cholesky decomposition
- `eigvals`, `eigvalsh` — eigenvalues
- `eigvecs`, `eigvecsh` — eigenvectors

</details>

<details>
<summary><b>🔍 Indexing & Gathering</b></summary>

- `gather` — multi-dimensional index gathering
- `scatter` — multi-dimensional scatter/updates
- `iota n` — generate sequence `[0, 1, ..., n-1]`

</details>

<details>
<summary><b>🎛️ Control Flow</b></summary>

- `choice cond x y` — conditional selection (`where`)
- `sorted` — sort along axis
- `fori_loop fn n init aux` — counted loop with state

</details>

<details>
<summary><b>🎯 Constants</b></summary>

- `zeros`, `empty` — zero/empty tensors
- `ofNat n` — constant tensor with value `n`

</details>

---

## 🎓 The `xla with` DSL

```lean
xla with
  arg1 : dtype [shape],    -- argument declarations
  arg2 : dtype [shape]
returns
  dtype [shape]            -- return type
begin
  let_expr name : dtype [shape] := expr;   -- bind intermediate values
  let name := expr;                        -- generic let binding
  fori_loop fn, count, init, aux           -- counted loop
  return expr                              -- return expression
```

**DTypes:** `float` | `int`

**Shapes:** List of naturals, e.g., `[n, m, 32]`

---

## 🏗️ Project Structure

```
JaxProof/
├── JaxProof/
│   ├── Expr.lean      📝 XLA AST & code generation
│   ├── Eval.lean      ⚙️  Evaluation framework
│   ├── Impl.lean      🔧 FloatAsReal implementation
│   ├── Lib.lean       📦 Helper functions
│   └── Tensor.lean    🔲 Tensor type & operations
├── Test/              ✅ Example proofs
│   ├── matmul.lean
│   ├── fori_loop.lean
│   ├── normalize.lean
│   ├── scatter.lean
│   └── index_nonzero.lean
└── README.md          📖 This file
```

---

## 🔮 Roadmap

- [x] Core XLA expression DSL
- [x] Code generation (XLA IR)
- [x] `FloatAsReal` interpreter with proofs
- [x] 40+ XLA primitives
- [ ] Interval arithmetic mode (for numerical error bounds)
- [ ] Probabilistic interpretation (for randomized algorithms)
- [ ] Direct JAX/Python bridge for hardware execution

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug fixes
- ✨ New XLA primitives
- 📖 Documentation improvements
- 🧪 Additional examples and proofs

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[⭐ Star this repo](https://github.com/yourusername/JaxProof)** if you find it useful!

Built with ❤️ using [Lean 4](https://lean-lang.org/)

</div>
$                                                                                                                                                                                                                                                             
Tip: press Ctrl-D or send 'exit' to quit
$                                                                                                                                                                                                                                                             
Bye!
root@183e312c349c:~/JaxProof# cat README.md 
# 🔬 JaxProof

> **Verified XLA Code Generation from Lean 4**

[![Lean](https://img.shields.io/badge/Lean-4-blue)](https://lean-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Write mathematically correct GPU/TPU code. Prove it before running it. Never waste GPU hours on bugs again.

---

## ✨ What is JaxProof?

**JaxProof** is a formal verification framework that bridges the gap between mathematical specifications and high-performance XLA (Accelerated Linear Algebra) code. It allows you to:

- 📝 Write XLA programs in Lean 4 using an elegant DSL
- 🎯 Generate XLA intermediate representation (IR)
- ✅ Formally prove that your code matches mathematical specifications
- 🚀 Run with confidence on JAX/XLA backends

---

## 🚀 Quick Start

### Matrix Multiplication

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

-- Generate XLA IR
#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).pretty_print
```

Output:
```
%0: transpose [1, 0] $0
%1: dot_general 1 0 %0 $1
return %1, 
```

### Prove Correctness

```lean
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Jax.FloatAsReal *[x, y] = *[x * y] := by
  simp [matmul, Jax.ExprGroup.eval]
  apply funext; intro i
  apply funext; intro j
  conv_lhs => change (∑ k, fun i j => x i k * y k j) i j
  simp [Matrix.mul_apply, Finset.sum_apply]
```

💡 **The proof guarantees** that the generated XLA code computes the exact mathematical matrix product.

---

## 📚 Examples Gallery

### 🔁 Counted Loops with `fori_loop`

Square a vector `m` times:

```lean
def power_loop {n : ℕ} :=
  xla with
    x : float [n],
    m : int []
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
    fori_loop loop_fn, m, (.cons x .nil), .nil
```

**Generated IR:**
```
return fori_loop($1, , @0, ($0, ), ())
@0:
%0: mul $1 $1
returns %0, 
```

**Theorem:** `(power_loop.eval Jax.FloatAsReal) *[x, m] = *[x ^ (2 ^ m)]`

---

### 📐 Vector Normalization

Compute L2 norm and normalize a vector:

```lean
def norm_xla {n : ℕ} :=
  xla with
    x : float [n]
  returns
    float []
  begin
    let_expr x2 : float [n] := .bind .mul *[x, x];
    let_expr x2_sumed : float [] := .bind (.sum 1) *[x2];
    return .bind .sqrt *[x2_sumed]

def normalize_xla {n : ℕ} :=
  let f₀ := Jax.ExprGroup.cons (Jax.Expr.arg 0) (norm_xla (n := n))
  let f₁ :=
    xla with
      x : float [n],
      norm_x : float []
    returns
      float [n]
    begin
      let_expr norm_x : float [n] := .bind (.broadcast [(n, false)]) *[norm_x];
      return .bind .div *[x, norm_x]
  Jax.ExprGroup.apply f₀ f₁
```

**Proof:** `normalize_xla.eval Jax.FloatAsReal *[x] = *[x / ‖x‖₂]`

---

### 🎯 Inverse Permutation with Scatter

```lean
def inv_permutation {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    return .bind .scatter *[0, Jax.iota, x]
```

If `x` represents a permutation `σ`, this computes `σ⁻¹`.

---

### 🔍 Index of Nonzero Elements

```lean
def idxOfNonzero {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    let_expr mask : int [n] := Jax.choice x 1 0;      -- 1 where x≠0, else 0
    let_expr counts : int [n] := Jax.cumsum mask;     -- running count
    let_expr result : int [n] := Jax.choice x counts 0; -- keep counts where x≠0
    return result
```

---

## 🧮 Supported XLA Operations

<details>
<summary><b>⚡ Element-wise Operations</b></summary>

| Unary | Binary |
|-------|--------|
| `abs`, `neg`, `sqrt`, `cbrt` | `add`, `sub`, `mul`, `div` |
| `exp`, `exp2`, `expm1` | `div_int`, `mod` |
| `cos`, `sin`, `acos`, `asin` | `and`, `eq` |
| `cosh`, `sinh`, `acosh`, `asinh` | |
| `atan`, `atanh` | |
| `erf`, `erfc`, `erf_inv` | |
| `bessel_i0e`, `bessel_i1e` | |
| `ceil` | |

</details>

<details>
<summary><b>📊 Reduction & Scan</b></summary>

- `sum n` — sum over leading `n` dimensions
- `argmax axis`, `argmin axis` — index of max/min
- `cumsum`, `cummax`, `cummin`, `cumprod`, `cumlogsumexp`

</details>

<details>
<summary><b>🔄 Shape Manipulation</b></summary>

- `transpose perm` — arbitrary dimension permutation
- `broadcast dims` — expand dimensions with broadcasting
- `concat axis` — concatenate along any axis
- `dynamic_slice`, `dynamic_update_slice`

</details>

<details>
<summary><b>🔢 Linear Algebra</b></summary>

- `dot_general batch contract lhs rhs` — generalized dot product
- `cholesky` — Cholesky decomposition
- `eigvals`, `eigvalsh` — eigenvalues
- `eigvecs`, `eigvecsh` — eigenvectors

</details>

<details>
<summary><b>🔍 Indexing & Gathering</b></summary>

- `gather` — multi-dimensional index gathering
- `scatter` — multi-dimensional scatter/updates
- `iota n` — generate sequence `[0, 1, ..., n-1]`

</details>

<details>
<summary><b>🎛️ Control Flow</b></summary>

- `choice cond x y` — conditional selection (`where`)
- `sorted` — sort along axis
- `fori_loop fn n init aux` — counted loop with state

</details>

<details>
<summary><b>🎯 Constants</b></summary>

- `zeros`, `empty` — zero/empty tensors
- `ofNat n` — constant tensor with value `n`

</details>

---

## 🎓 The `xla with` DSL

```lean
xla with
  arg1 : dtype [shape],    -- argument declarations
  arg2 : dtype [shape]
returns
  dtype [shape]            -- return type
begin
  let_expr name : dtype [shape] := expr;   -- bind intermediate values
  let name := expr;                        -- generic let binding
  fori_loop fn, count, init, aux           -- counted loop
  return expr                              -- return expression
```

**DTypes:** `float` | `int`

**Shapes:** List of naturals, e.g., `[n, m, 32]`

---

## 🏗️ Project Structure

```
JaxProof/
├── JaxProof/
│   ├── Expr.lean      📝 XLA AST & code generation
│   ├── Eval.lean      ⚙️  Evaluation framework
│   ├── Impl.lean      🔧 FloatAsReal implementation
│   ├── Lib.lean       📦 Helper functions
│   └── Tensor.lean    🔲 Tensor type & operations
├── Test/              ✅ Example proofs
│   ├── matmul.lean
│   ├── fori_loop.lean
│   ├── normalize.lean
│   ├── scatter.lean
│   └── index_nonzero.lean
└── README.md          📖 This file
```

---

## 🔮 Roadmap

- [x] Core XLA expression DSL
- [x] Code generation (XLA IR)
- [x] `FloatAsReal` interpreter with proofs
- [x] 40+ XLA primitives
- [ ] Interval arithmetic mode (for numerical error bounds)
- [ ] Probabilistic interpretation (for randomized algorithms)
- [ ] Direct JAX/Python bridge for hardware execution

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug fixes
- ✨ New XLA primitives
- 📖 Documentation improvements
- 🧪 Additional examples and proofs

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[⭐ Star this repo](https://github.com/yourusername/JaxProof)** if you find it useful!

Built with ❤️ using [Lean 4](https://lean-lang.org/)

</div>
