# 🔬 JaxProof

> **Verified XLA Code Generation from Lean 4**

[![Lean](https://img.shields.io/badge/Lean-4-blue)](https://lean-lang.org/)

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
import SSA

def matmul {n m l : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n, m]⟩,
    y : ⟨.float, [m, l]⟩
  begin
    let_expr x : [⟨.float, [m, n]⟩] := Xla.transpose [0,1].formPerm x;
    let_expr z : [⟨.float, [n, l]⟩] := Xla.dot_general [] [m] [n] [l] x y;
    return z

-- Generate XLA IR
#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code
```

Output:
```
%0 = transpose [1, 0];;$0
%1 = dot_general 1 0;;%0;$1
return %1
```

### Prove Correctness

```lean
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval Xla.DirectImpl x y = Index.single (x * y) := by
  simp [matmul, Xla.dot_general, Xla.bindPrim, SSA.Expr.eval]
  congr
  ext i j
  simp [Matrix.mul_apply]
```

💡 **The proof guarantees** that the generated XLA code computes the exact mathematical matrix product.

---

## 📚 Examples Gallery

### 🔁 Counted Loops with `fori_loop`

Square a vector `m` times:

```lean
def power_loop {n : ℕ} :=
  let body :=
    ssa Xla.XlaOp with
      x : ⟨.float, [n]⟩
    begin
      return Xla.mul x x
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩,
    m : ⟨.int, []⟩
  begin
    return Xla.fori_loop body m x .nil
```

**Generated IR:**
```
%0 = repeat;&0;$1;$0
return %0

&0
%0 = mul;;$0;$0
return %0
```

**Theorem:** `(power_loop (n := n)).eval Xla.DirectImpl x (m : ℤ) = Index.single (x ^ (2 ^ m))`

---

### 📐 Vector Normalization

Compute L2 norm and normalize a vector:

```lean
def norm {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let_expr x2 : [⟨.float, [n]⟩] := Xla.mul x x;
    let_expr x2_sumed : [⟨.float, []⟩] := Xla.bindPrim (.sum 1) x2;
    return Xla.bindPrim .sqrt x2_sumed

def normalize {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let_expr nrm : [⟨.float, []⟩] := norm (n := n);
    let_expr nrm_bc : [⟨.float, [n]⟩] := Xla.bindPrim (.broadcast [(n, false)]) nrm;
    return Xla.bindPrim .div (x.append nrm_bc)
```

**Proof:** `normalize.eval Xla.DirectImpl x = Index.single (fun i => x i / √(∑ j, (x j)²))`

---

### 🎯 Inverse Permutation with Scatter

```lean
def inv_permutation {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.int, [n]⟩
  begin
    return Xla.scatter (s := [n]) 0 Xla.iota x
```

If `x` represents a permutation `σ`, this computes `σ⁻¹`.

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
- `fori_loop body n init aux` — counted loop with carried and auxiliary state

</details>

<details>
<summary><b>🎯 Constants</b></summary>

- `zeros`, `empty` — zero/empty tensors
- `ofNat n` — constant tensor with value `n`

</details>

---

## 🎓 The `ssa` DSL

```lean
ssa XlaOp with
  arg1 : dtype [shape],    -- argument declarations
  arg2 : dtype [shape]
begin
  let_expr name : dtype [shape] := expr;   -- bind intermediate values
  let name := expr;                        -- generic let binding
  return expr                              -- return expression
```

**DTypes:** `float` | `int`

**Shapes:** List of naturals, e.g., `[n, m, 32]`

---

## 🏗️ Project Structure

```
SSA/
├── SSA/
│   ├── Core.lean      📝 SSA AST & code generation
│   ├── Curry.lean     🔗 Curried tensor types
│   ├── Tensor.lean    🔲 Tensor operations
│   ├── Xla/
│   │   ├── Op.lean    ⚡ XLA primitive ops
│   │   ├── Impl.lean  🔧 DirectImpl semantics
│   │   └── Libs.lean  📦 Helpers & DSL sugar
│   └── ...
├── Tests/             ✅ Example proofs
│   ├── matmul.lean
│   ├── fori_loop.lean
│   └── scatter.lean
├── README.md          📖 This file
└── lakefile.toml      📦 Lean package config
```

---

## 🔮 Roadmap

- [x] Core XLA expression DSL
- [x] Code generation (XLA IR)
- [x] `DirectImpl` interpreter with proofs
- [x] `fori_loop` higher-order primitive
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

MIT License

---

<div align="center">

Built with ❤️ using [Lean 4](https://lean-lang.org/)

</div>
