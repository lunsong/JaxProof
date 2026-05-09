# SSA — Verified XLA for Scientific Computing

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

SSA replaces testing with proof. Suppose we want to develop a force field:

1. **Specification**: The human writes the desired properties of the function: equivarient, smoothness...
2. **Generation**: An LLM writes a function using `SSA` and *prove* that it satisfies the properties.
3. **Evaluation**: `SSA` generates compact IR ready to evaluate.

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
| <pre>def permInv {n : ℕ} :=<br>  ssa Xla.XlaOp with<br>    x : ⟨.int, [n]⟩<br>  begin<br>    return Xla.scatter (s := [n]) 0 Xla.iota x</pre> | <pre>%0 = const int [10] 0; <br>%1 = iota 10; <br>%2 = scatter; %0, %1, $0<br>return %2</pre> | $\text{permInv} \,\sigma = \sigma^{-1}$ |
| <pre>def idxOfNonzero {n : ℕ} :=<br>  ssa Xla.XlaOp with<br>    x : ⟨.int, [n]⟩<br>  begin<br>    let_expr x_nonzero : [⟨.int, [n]⟩] := Xla.choice x 1 0;<br>    let x_id := Xla.cumsum x_nonzero;<br>    return Xla.choice x x_id 0</pre> | <pre>%0 = const int [12] 1; <br>%1 = const int [12] 0; <br>%2 = where; $0, %0, %1<br>%3 = cumsum; %2<br>%4 = where; $0, %3, %1<br>return %4</pre> | `idxOfNonzero` assign each nonzero element a unique index |
| <pre>def power_loop {n : ℕ} :=<br>  let body :=<br>    ssa Xla.XlaOp with<br>      x : ⟨.float, [n]⟩<br>    begin<br>      return Xla.mul x x<br>  ssa Xla.XlaOp with<br>    x : ⟨.float, [n]⟩,<br>    m : ⟨.int, []⟩<br>  begin<br>    return Xla.fori_loop body m x .nil</pre> | <pre>%0 = repeat; @0, $1, $0<br>return %0<br><br>&0<br>%0 = mul; $0, $0<br>return %0</pre> | $\text{power\_loop}(x, m) = x^{2^m}$ |

Each theorem states that the generated XLA code computes *exactly* the mathematical specification. The full proofs can be found in `Tests/scatter.lean`, `Tests/idxOfNonzero.lean`, and `Tests/fori_loop.lean`.

---

## 🐍 JAX Evaluator

A Python-based evaluator in `python/jax_eval.py` can execute the generated XLA IR using JAX. This provides a practical way to run verified kernels on GPU/TPU hardware.

```python
from python.jax_eval import evaluate
import jax.numpy as jnp

# Load IR generated from Lean
ir = """%0 = const int [12] 1; 
%1 = const int [12] 0; 
%2 = where; $0, %0, %1
%3 = cumsum; %2
%4 = where; $0, %3, %1
return %4"""

x = jnp.array([0, 5, 0, 3, 0, 0, 7, 0, 1, 0, 0, 2], dtype=jnp.int32)
result = evaluate(ir, x)
# result: [0, 1, 0, 2, 0, 0, 3, 0, 4, 0, 0, 5]
```

Supported operations include: `const`, `zeros`, `iota`, `add`, `sub`, `mul`, `div`, `neg`, `sqrt`, `sin`, `cos`, `exp`, `log`, `tanh`, `ceil`, `floor`, `sum`, `cumsum`, `transpose`, `broadcast`, `dot_general`, `where`, `scatter`, `gather`, `concat`, `sorted`, `repeat` (fori_loop), and `call` (function application).

Run the test suite:
```bash
python python/test_jax_eval.py
```

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
├── python/            🐍 JAX evaluator for generated IR
│   ├── jax_eval.py
│   └── test_jax_eval.py
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
