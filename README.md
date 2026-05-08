# SSA — Verified XLA for Scientific Computing

> **When AI writes your GPU kernels, how do you know they're correct?**

[![Lean](https://img.shields.io/badge/Lean-4-blue)](https://lean-lang.org/)

AI can now generate XLA kernels for molecular dynamics, Monte Carlo sampling, and neural-network-potential training. The code compiles, converges, and looks reasonable — yet a single transposed dimension in a scatter index, a broadcast mismatch in a force kernel, or an off-by-one in a reweighting loop corrupts the physics without ever raising an error.

**SSA** is a formal verification framework that bridges AI-generated high-performance code and mathematical proof. Write the specification in Lean 4, generate XLA intermediate representation, and obtain a machine-checked guarantee that the compiled kernel computes *exactly* what the mathematics demands — not approximately, not on a few test cases, but by proof.

---

## Why Tests Are Not Enough

In scientific computing, three properties are essential and fundamentally untestable:

| Property | Can you unit-test it? | Why not? |
|----------|----------------------|----------|
| **Continuity** | ❌ | A discontinuity lies at a measure-zero boundary; every pointwise test passes, yet the kernel injects an unphysical force spike. |
| **Gradient correctness** | ❌ | Finite-difference checks are too expensive for large models and miss discontinuities. A transposed Jacobian or missing chain-rule term produces forces that are not the gradient of anything. |
| **Stochastic convergence** | ❌ | Systematic bias from an incorrect transition kernel or reweighting formula is hidden under sampling noise. Every pointwise sample looks reasonable, yet the ensemble converges to the wrong distribution. |

Regression tests only catch bugs you have already seen. Code review does not scale with AI-generated volume. Formal proof is the only verification strategy that reasons about *all* inputs, all configurations, and all sample paths.

---

## How We Solve It

SSA replaces testing with proof. The workflow is simple:

1. **Specification**: The human writes the mathematical object — a Hamiltonian, a loss function, a reweighting formula, a transition kernel.
2. **Generation**: An LLM or DSL writes the XLA kernel.
3. **Verification**: Lean 4 proves the generated kernel computes *exactly* the specified object.

The proof catches transposed dimensions, broadcast mismatches, scatter mis-indexing, and formula errors that no test suite could find.

### Not Tied to XLA

The framework is designed for flexibility. The core SSA expression language is generic over operations and data types. You can:

- **Add new primitives easily.** An ODE solver, a stochastic differential equation integrator, or a custom reweighting kernel can be introduced as a new primitive operation with its own code-generation rule and mathematical semantics.
- **Swap the semantic model.** The same expression can be evaluated under different interpretations: model floats as real numbers (`DirectImpl`) for exact mathematical proofs, as intervals for rigorous error bounds, or as random variables for stochastic correctness proofs.
- **Target different backends.** The current backend generates XLA IR, but the code generator is pluggable. The same verified expression can be retargeted to LLVM, CUDA, or a custom DSL.

This means the framework grows with your needs. Today you verify deterministic force kernels. Tomorrow you add a `solve_ode` primitive and prove your MD integrator conserves energy. The next day you add a probabilistic interpretation and prove your Monte Carlo sampler targets the Boltzmann distribution — all within the same expression language.

---

## Gallery

| Lean definition | Generated code | Proved theorem |
|-----------------|----------------|----------------|
| <pre>def permInv {n : ℕ} :=<br>  ssa Xla.XlaOp with<br>    x : ⟨.int, [n]⟩<br>  begin<br>    return Xla.scatter (s := [n]) 0 Xla.iota x</pre> | <pre>%0 = const int [10] 0;;<br>%1 = iota 10;;<br>%2 = scatter;;%0;%1;$0<br>return %2</pre> | `permInv.eval Xla.DirectImpl (fun i => σ i) = Index.single (fun i => (σ.symm i : ℤ))` |
| <pre>def idxOfNonzero {n : ℕ} :=<br>  ssa Xla.XlaOp with<br>    x : ⟨.int, [n]⟩<br>  begin<br>    let_expr x_nonzero : [⟨.int, [n]⟩] := Xla.choice x 1 0;<br>    let x_id := Xla.cumsum x_nonzero;<br>    return Xla.choice x x_id 0</pre> | <pre>%0 = const int [12] 1;;<br>%1 = const int [12] 0;;<br>%2 = where;;$0;%0;%1<br>%3 = cumsum;;%2<br>%4 = where;;$0;%3;%1<br>return %4</pre> | `idxOfNonzero.eval Xla.DirectImpl x = Index.single (idxOfNonzero_def x)` |
| <pre>def power_loop {n : ℕ} :=<br>  let body :=<br>    ssa Xla.XlaOp with<br>      x : ⟨.float, [n]⟩<br>    begin<br>      return Xla.mul x x<br>  ssa Xla.XlaOp with<br>    x : ⟨.float, [n]⟩,<br>    m : ⟨.int, []⟩<br>  begin<br>    return Xla.fori_loop body m x .nil</pre> | <pre>%0 = repeat;&0;$1;$0<br>return %0<br><br>&0<br>%0 = mul;;$0;$0<br>return %0</pre> | `(power_loop (n := n)).eval Xla.DirectImpl x (m : ℤ) = Index.single (x ^ (2 ^ m))` |

Each theorem states that the generated XLA code computes *exactly* the mathematical specification. The full proofs can be found in `Tests/scatter.lean`, `Tests/idxOfNonzero.lean`, and `Tests/fori_loop.lean`.

---

## 🧮 Supported Operations

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

## 🔮 Roadmap: Verified AI-Generated Simulation Code

We are building toward a world where **every GPU kernel in an AI-generated simulation pipeline carries a machine-checked proof of correctness.**

| Milestone | Status | Impact |
|-----------|--------|--------|
| Core XLA expression DSL | ✅ Done | Write and verify force fields, descriptors, losses |
| Code generation (XLA IR) | ✅ Done | JIT to GPU with zero overhead |
| `DirectImpl` interpreter with proofs | ✅ Done | Prove kernels compute exact forces/energies/observables |
| `fori_loop` higher-order primitive | ✅ Done | Verified iterative solvers and training updates |
| 40+ XLA primitives | ✅ Done | Full coverage of dense linear algebra |
| **Stochastic semantics** | 🔄 Next | **Prove MC samplers and reweighting kernels target exact distributions** |
| **Gradient verification** | 🔄 Next | **Prove forces are exact gradients of the learned potential** |
| **ODE solver primitive** | 🔄 Next | **Prove MD integrators conserve symplectic structure / energy** |
| **Training loop verification** | 🔄 Next | **Prove loss kernels compute exact force-matching / energy-matching objectives** |
| Interval arithmetic | 📋 Planned | Rigorous floating-point error bounds for long trajectories |
| Direct JAX/Python bridge | 📋 Planned | Drop-in replacement for `jax.jit` with proof certificates |

---

## 🏗️ Project Structure

```
SSA/
├── SSA/
│   ├── Core.lean      📝 SSA AST, evaluation, and code generation
│   ├── Curry.lean     🔗 Curried tensor types for arbitrary rank
│   ├── Tensor.lean    🔲 Tensor operations (sum, broadcast, einsum, ...)
│   ├── Xla/
│   │   ├── Op.lean    ⚡ XLA primitive operations (40+ ops)
│   │   ├── Impl.lean  🔧 DirectImpl: mathematical semantics
│   │   └── Libs.lean  📦 DSL helpers and sugar
│   └── ...
├── Tests/             ✅ Machine-checked proof examples
│   ├── matmul.lean
│   ├── fori_loop.lean
│   ├── idxOfNonzero.lean
│   ├── scatter.lean
│   └── normalize.lean
├── README.md          📖 This file
└── lakefile.toml      📦 Lean package configuration
```

---

## 🤝 Contributing

We welcome contributions across the full stack:
- 🐛 Bug fixes in primitives or proofs
- ✨ New XLA primitives (especially MD-relevant ones: Ewald summation, PPPM, spherical harmonics, ...)
- 🧪 Additional examples from molecular simulation and ML-potential training pipelines
- 📖 Documentation and tutorials for the chemistry/physics community

---

## 📄 License

MIT License

---

<div align="center">

Built with ❤️ using [Lean 4](https://lean-lang.org/)

</div>
