# JaxProof

[![Lean Action CI](https://github.com/lunsong/JaxProof/actions/workflows/lean_action_ci.yml/badge.svg)](https://github.com/lunsong/JaxProof/actions/workflows/lean_action_ci.yml)
[![Lean v4.33.1](https://img.shields.io/badge/Lean-v4.33.1-blueviolet)](https://lean-lang.org)

**Write a tensor program once. Prove what it computes. Run it on JAX.**

JaxProof is an SSA/XLA-style tensor DSL embedded in Lean 4. A program is an ordinary
Lean definition, and that same definition gives you three things:

| | |
|---|---|
| **Math** | a real-number semantics (`Expr.eval`) you can state and prove theorems about |
| **IR** | XLA-like SSA text (`Expr.code`) with structural deduplication, ready for a compiler |
| **Execution** | a JAX evaluator (`python/eval.py`) that runs the emitted IR, matching the Lean semantics |

So the object you prove things about is the object you run — no transcription between
the spec and the kernel.

## One program, three guarantees

From [`Examples/matmul.lean`](Examples/matmul.lean), the program definition:

```lean
import Xla

def matmul {n m l : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n, m]⟩, ⟨.float, [m, l]⟩]
    ⟨.float, [n, l]⟩ :=
  Soir.Expr.ofFn fun x y =>
    let x := Xla.transpose x [0, 1].formPerm;
    Xla.dot_general [] [m] [n] [l] x y
```

**Emits IR** — `#eval IO.println (matmul (n:=10) (m:=20) (l:=30)).code` prints
(verbatim):

```text
%0 = transpose [1, 0]; $0
%1 = dot_general 1 0; %0, $1
return %1
```

**Proves a theorem** — the full proof stays short:

```lean
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval x y = x * y := by
  ext i j
  simp [matmul, reduce_xla, reduce_soir, reduce_tensor]
  congr
  erw [Finset.sum_apply, Finset.sum_apply]
  rfl
```

**Runs in JAX** — `python/eval.py` parses the IR and reproduces the `DirectImpl`
semantics, so the emitted program evaluates to `x @ y` under JAX.

## Quick start

Requires [elan](https://github.com/leanprover/elan) (Lean is pinned by
`lean-toolchain`).

```bash
lake exe cache get                       # fetch Mathlib oleans (recommended)
lake build                               # default target: Soir
lake build Soir Xla Random Examples QMC  # everything

lake env lean Examples/matmul.lean       # elaborate a file; `#eval` prints IR
```

Python side (`jax`, `numpy`):

```bash
python3 python/test_e2e.py               # emitted IR vs. numpy references
python3 python/test_ferminet.py          # FermiNet amplitudes under JAX
```

## What's in the box

| Path | Contents |
|---|---|
| [`Soir/`](Soir/) | Core DSL: `Expr` (multi-in/multi-out SSA), `Curry`/`Index` glue, `Impl` semantics, IR codegen, `reduce_soir` simp set |
| [`Xla/`](Xla/) | Tensor dialect: `TensorType`, the `XlaPrimOp`/`XlaHigherOp` op set (`vmap`, `repeat`), `DirectImpl` real-number semantics, `Tensor` library |
| [`Random/`](Random/) | Probabilistic dialect: key-indexed draws, `RandVar` semantics as measurable functions, expectation/independence lemmas |
| [`QMC/`](QMC/) | `IsValidQMCWavefunction` contract and two neural-network ansätze: FermiNet, PauliNet |
| [`Examples/`](Examples/) | 13 programs; 10 fully proved |
| [`python/`](python/) | JAX evaluator for emitted IR, Lean→IR bridge, test suite |
| [`web_service/`](web_service/) | Flask service: natural language → Lean theorem + DSL program via an LLM, then builds it |

## Why it's interesting

- **Proofs are cheap.** The `reduce_xla` / `reduce_soir` / `reduce_tensor` simp sets
  evaluate a program's `Curry`/`Index`/`Tensor` scaffolding definitionally, so a
  correctness proof is often a handful of lines of `simp` (see the gallery below).
- **Codegen stays fast.** Keeping components as closed libraries composed with
  `Expr.apply` turns a quadratic hash-lookup blowup into near-linear work: the H₂
  FermiNet program went from 17 s to 0.4 s of codegen, PauliNet from 1 m 53 s to
  4.3 s, with the two styles evaluating identically in the JAX harness.
- **Smoothness is compositional.** `contDiff_eval` (a head-directed
  `continuity`-style tactic) composes per-primitive `C²` lemmas along a program's
  dataflow — that is what discharges the smoothness field of the QMC contract.
- **The LLM front end is formal, not vibes.** `web_service/` asks a model for a
  Lean theorem *and* a DSL implementation, then makes `lake env lean` the judge,
  iterating on real compiler errors.

## Example gallery

| File | Program | Status |
|---|---|---|
| [`matmul.lean`](Examples/matmul.lean) | matrix multiplication via `dot_general` | proved |
| [`attention.lean`](Examples/attention.lean) | `softmax` and scaled dot-product attention | proved |
| [`normalize.lean`](Examples/normalize.lean) | Euclidean norm and normalization | proved |
| [`transpose.lean`](Examples/transpose.lean) | transposition and transposed sums | proved |
| [`Coulomb.lean`](Examples/Coulomb.lean) | pairwise `1/‖xᵢ-xⱼ‖` electron repulsion | proved |
| [`scatter.lean`](Examples/scatter.lean) | permutation inverse via `scatter` | proved |
| [`fori_loop.lean`](Examples/fori_loop.lean) | `x ↦ x^(2^m)` via the `repeat` higher-order op | proved |
| [`offDiag.lean`](Examples/offDiag.lean) | drop the diagonal of a matrix | proved |
| [`idxOfNonzero.lean`](Examples/idxOfNonzero.lean) | rank of a non-zero entry (sort/iota) | proved |
| [`random.lean`](Examples/random.lean) | sampling, unbiased estimators, martingale identities | proved |
| [`spmm.lean`](Examples/spmm.lean), [`denseToCoo.lean`](Examples/denseToCoo.lean), [`cooToDense.lean`](Examples/cooToDense.lean) | sparse matmul, dense↔COO | `sorry`s remain |

<details>
<summary><b>The core DSL, in 60 seconds</b></summary>

- `Expr op args outs` is a value in SSA form: `nil`, `append`, `select`, `arg`,
  `apply` (a library call), `bind` (a primitive op with recursive inputs). Programs
  are built with `Soir.Expr.ofFn`, a curried function over the formal arguments.
- `Xla.SimpleExpr.eval` interprets a program as a function on real tensors via the
  `DirectImpl` instance of `Impl` (float entries `ℝ`, int entries `ℤ`).
- `Expr.code` emits textual SSA (`%0 = add; $0, $1`), hoisting repeated
  sub-expressions into numbered libraries `@0, @1, ...`. Deduplication uses a
  structural hash (`Expr.hashExpr`), so codegen is reproducible across builds.

</details>

## QMC: a validity contract for learned wavefunctions

[`QMC/Ansatz.lean`](QMC/Ansatz.lean) defines

```lean
structure IsValidQMCWavefunction (Ψ : Wavefunction n₁ n₂) : Prop where
  antisymmetric : IsAntisymmetric Ψ
  smooth        : ContDiff ℝ 2 (Function.uncurry Ψ)
  exp_decay     : ∃ C k, 0 < k ∧ ∀ x, |Ψ x| ≤ C * exp (-k * ‖x‖)
```

These are the properties whose violation would be a *silent* cheat for variational
Monte Carlo (bosonic collapse; automatic differentiation dropping distributional
deltas), plus one normalizability certificate that is cheaper to prove compositionally
than to monitor heuristically. `Ansatz.isValid` quantifies over all parameters and all
molecules, so every positive quantity is reparameterized with `exp` inside the ansatz.

- [`QMC/FermiNet.lean`](QMC/FermiNet.lean) implements FermiNet as a closed-library
  `Xla.SimpleExpr` and proves `smooth` with `contDiff_eval`; `antisymmetric` and
  `exp_decay` are sketched in the docstring.
- [`QMC/PauliNet.lean`](QMC/PauliNet.lean) implements a SchNet-style PauliNet whose
  parameter count is independent of molecule size; the contract fields are argued in
  the docstring.
- `python/test_ferminet.py` checks the emitted programs end-to-end under JAX: finite
  amplitudes, antisymmetry in each spin sector, exponential decay.

## Smoothness and the `contDiff_eval` tactic

[`Xla/Smooth.lean`](Xla/Smooth.lean) proves that each primitive's `DirectImpl`
semantics is `C²` in a normed `ℝ`-vector space of inputs, and
[`Xla/SmoothTactic.lean`](Xla/SmoothTactic.lean) composes those leaf lemmas
(`@[contDiff_eval_rule]`) along a program's dataflow. The tactic dispatches on the
head of the goal's body, costs one `apply` per program node, and leaves
positivity/non-vanishing side conditions to the caller.

## Random: a probabilistic dialect

[`Random/Impl.lean`](Random/Impl.lean) interprets programs as random variables: the
model is `Measure.infinitePi` over independent `normal`/`uniform` draws, with
`RandVar.mean` the expectation. Draws are addressed by keys and `shuffle k = k + 1`
derives a fresh key, so `normal k` and `normal (shuffle k)` are independent.
[`Examples/random.lean`](Examples/random.lean) proves the textbook facts: means of
the primitive draws, unbiasedness of sample averages, `E[Z²] = 1`, products of
independent draws, and a martingale/noise step.

## Python tooling

```python
import sys; sys.path.insert(0, "python")
import ops  # registers the op handlers
from eval import evaluate

ir = """%0 = const int [12] 1;
%1 = const int [12] 0;
%2 = where; $0, %0, %1
%3 = cumsum; %2
%4 = where; $0, %3, %1
return %4"""

evaluate(ir, my_int_array)   # under JAX, matching DirectImpl semantics
```

- `python/eval.py` — IR parser + registry-based evaluator.
- `python/ops.py` — the op handlers.
- `python/lean_eval.py` — the other direction:
  `eval("Examples.offDiag", x, params=[("n", 4)])` builds the Lean declaration,
  extracts its IR, and evaluates it.

Tests: [`test_eval.py`](python/test_eval.py) (parser/semantics),
[`test_jit.py`](python/test_jit.py) (jit/vmap coverage),
[`test_e2e.py`](python/test_e2e.py) (Lean-emitted IR vs. numpy),
[`test_lean_eval.py`](python/test_lean_eval.py) (bridge),
[`test_ferminet.py`](python/test_ferminet.py) (FermiNet end-to-end).

## Web service

A two-step LLM workflow: natural language → Lean theorem statement → DSL program +
proof, with iterative build-fix against `lake env lean` and IR extraction.

```bash
cd web_service
python3 -m venv venv
venv/bin/pip install flask flask-cors openai
export OPENAI_API_KEY=sk-...          # or create web_service/keys
./run.sh                              # http://localhost:5000
```
