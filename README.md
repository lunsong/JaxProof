# JaxProof

**Write programs in Lean. Run them with GPU. Prove them correct.**

[![Lean Action CI](https://github.com/lunsong/JaxProof/actions/workflows/lean_action_ci.yml/badge.svg)](https://github.com/lunsong/JaxProof/actions/workflows/lean_action_ci.yml)
`Lean v4.33.1` · `mathlib v4.33.1`

JaxProof is a Lean 4 framework for *programming, executing, and verifying* programs.
Its core is **Soir** (Second-Order IR), a dependently typed SSA representation that is
*dialect-generic*: a program is `Soir.Expr op args outs` with with primitive ops `op : OpType data`
and some data type system `data`. Soir itself is not tied to any domain — `Xla` dialect for vector programs,
`Random` dialect for stochastic programs, and others can be added. Each dialect provides

* a readable SSA IR (`Expr.code`, via the dialect's `ToString` instance),
* a mathematical semantics (`Expr.eval` with the dialect's `Impl` instance) that
  theorems are stated about,
* proof automation for that semantics — for the `Xla` dialect, rewrite sets for
  evaluation and a `contDiff_eval` tactic for smoothness, so verification composes
  along the program's dataflow.

---

## Why

Guarantees about programs come from different tools, and each property should get the
cheapest one that actually works for it. We use QMC as an example. Suppose we want to use
AI to find better QMC ansatz. A valid ansatz should have a few properties which can
be split into two cases

* **Numerically detectable properties are tested.** Antisymmetry of a wavefunction,
  for instance, can be checked directly in Python by evaluating the program on
  permuted inputs and asking for the sign.
* **Properties no numerical check can see must be proved.** Piecewise linear wavefunctions
  have vanishing local energy, thus we can construct a wavefunction whose QMC energy
  is arbitarily low.

JaxProof makes the proof side of that trade-off cheap and compositional, starting
from one program definition:

> **one program definition** → executable IR, a Lean semantics, and theorems
> relating the two (plus everything you prove *about* the semantics).

QMC is the clearest demonstration of this
trade-off in practice. An LLM generating an ansatz is scored by the variational energy,
so it is incentivized to be non-smooth — the loophole no test suite can catch, and the
one JaxProof turns into a proof obligation: `QMC/Ansatz.lean` states the contract, and
`QMC/` implements FermiNet and PauliNet against it.

## Highlights

* **Dependently typed, dialect-generic SSA core (`Soir`)** — `Expr op args outs`
  describes multi-input/multi-output expression graphs; the value type, shapes,
  dtypes, and arities are part of the type, so ill-posed programs do not elaborate.
* **Sharing-aware codegen** — `Expr.code` deduplicates sub-expressions by a structural
  hash and hoists shared subgraphs into `@n` library bodies. Writing components as
  closed libraries composed with `Expr.apply` keeps generation time flat in program
  size (FermiNet H₂: ~0.4 s vs. ~17 s for a flat program; `QMC/FermiNet.lean`).
* **Two matching semantics, one program** — for the `Xla` dialect, `Expr.eval`
  interprets a program into Lean (`DirectImpl`, tensors over `ℝ`/`ℤ`), while
  `python/ops.py` executes the emitted IR with JAX; the same declarations drive both,
  so theorem and runtime can drift only if a bug creeps into the backend.
* **Proof automation** — `reduce_soir`, `reduce_xla`, `reduce_tensor` simp sets turn
  semantic equations into ordinary `simp` goals, and `contDiff_eval`
  (`Xla/SmoothTactic.lean`) composes per-primitive C² lemmas along the dataflow.
* **Random dialect** — `Random.SimpleExpr` denotes a real random variable over an
  independent product law, with `RandVar.mean` and independence lemmas, so Monte
  Carlo estimators (unbiasedness, martingale steps) can be *proved*.
* **Worked example: a QMC validity contract** — `Ansatz.isValid` packages antisymmetry,
  C² smoothness, and an exponential envelope as one structure checked per ansatz, for
  all molecules and all parameter values.

## Dialects

`Soir/Core.lean` knows nothing about XLA or JAX: it defines `Expr`, `Expr.eval`, and
`Expr.code` parametrically. A dialect supplies four things:

| Piece | Interface |
|---|---|
| value type | `data : Type` (e.g. tensors, random variables) |
| op signatures | `op : OpType data`, built with the `SimpleOp`/`CombineOp` combinators |
| Lean semantics | an `Impl op impl` instance, i.e. `impl : data → Type` with a meaning per op |
| codegen | a `ToString (op exprs ins outs)` instance, rendering each op as an IR line |

The repository ships two dialects, and the same core serves both:

| Dialect | `data` | Semantics `impl` | Executed/proved by |
|---|---|---|---|
| `Xla` | `TensorType` (dtype × shape) | `DirectImpl`: `Tensor ℝ s` / `Tensor ℤ s` | SSA IR executed by JAX (`python/ops.py`); equation proofs via `reduce_xla`/`reduce_tensor`, smoothness via `contDiff_eval` |
| `Random` | `RandType` (`data`/`key`) | `RandType.impl`: `RandVar` / `ℕ` | measure-theoretic proofs via `reduce_random` (`mean`, independence) |

Adding a dialect (a new numeric semantics, a probability calculus, a cost model, ...)
means giving it these instances; the core and codegen need no changes (a runtime
backend, like the Python IR interpreter for `Xla`, is per-dialect).

## Quick start

Lean is managed by [`elan`](https://github.com/leanprover/elan); `lake` will fetch the
pinned toolchain and Mathlib automatically. The Python evaluator needs `jax` and
`numpy`.

```bash
git clone https://github.com/lunsong/JaxProof.git
cd JaxProof

# default target is Soir; build the rest explicitly
lake build Soir Xla Random Examples QMC

# elaborate an example and print its generated IR
lake env lean Examples/idxOfNonzero.lean
```

The last command prints:

```text
%0 = const int [12] 1;
%1 = const int [12] 0;
%2 = where; $0, %0, %1
%3 = cumsum; %2
%4 = zeros int [12];
%5 = where; $0, %3, %4
return %5
```

Warm builds are fast when Mathlib is cached (seconds for this repository); a cold
build compiles Mathlib and takes tens of minutes. CI
(`.github/workflows/lean_action_ci.yml`) builds the project on every push and pull
request.

## A tour: an `Xla` program, its IR, and a theorem

A program in the `Xla` dialect is built from `Soir.Expr.ofFn` (a curried lambda over
the inputs) and the smart constructors in `Xla/Libs.lean`. Here is matrix
multiplication:

```lean
import Xla

def matmul {n m l : ℕ} :
    Xla.SimpleExpr
      [⟨.float, [n, m]⟩, ⟨.float, [m, l]⟩]
      ⟨.float, [n, l]⟩ :=
  Soir.Expr.ofFn fun x y =>
    let x := Xla.transpose x perm[0, 1];
    Xla.dot_general [] [m] [n] [l] x y

-- emit the IR
#eval IO.println (matmul (n := 10) (m := 20) (l := 30)).code

-- and prove what the program computes
example (n m l : ℕ) (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    matmul.eval x y = x * y := by
  ext i j
  simp [matmul, reduce_xla, reduce_soir, reduce_tensor]
  congr
  erw [Finset.sum_apply, Finset.sum_apply]
  rfl
```

`SimpleExpr.eval` is the Lean semantics; `reduce_*` exposes the semantics of each node
so the proof is a composition of definitions and the lemmas in `Xla/Tensor.lean`.
`Examples/attention.lean` proves a full attention block (`softmax` + two `einsum`s)
equal to its textbook definition.

### Running the `Xla` program in JAX

`python/lean_eval.py` resolves a Lean declaration to its IR through the toolchain
(cached), and `python/eval.py` + `python/ops.py` evaluate the IR with JAX:

```bash
cd python
python3 -c "
import ops                        # register the XLA op handlers
from lean_eval import lean_eval   # Lean declaration -> IR -> JAX
import jax.numpy as jnp

x = jnp.arange(16.0).reshape(4, 4)
print(lean_eval('Examples.offDiag', x, params=[('n', 4)]))
"
# [[ 2.  3.  4.  5.]
#  [ 7.  8.  9. 10.]
#  [12. 13. 14. 15.]]
```

Pass `intermediates=[0, 3, ...]` to `evaluate` to capture `%0`, `%3`, ... for
debugging, and use `@register_op("name")` to extend the evaluator with new IR ops.

## Repository layout

`Soir/` is dialect-agnostic; `Xla/`, `Random/`, and `QMC/` are instantiations of it.

| Path | Contents |
|---|---|
| `Soir/Core.lean` | the dialect-generic SSA expression type, codegen (`Expr.code`), and the `Expr.eval` interpreter |
| `Soir/Curry.lean` | `Curry`/`Index`: dependently typed curried functions, the tensor representation used by the semantics |
| `Soir/Meta.lean` | the `reduce_soir` simp set and simproc registration |
| `Xla/Op.lean` | the `Xla` dialect: `DType`, `TensorType`, and the primitive op signature (~70 ops, plus `vmap`/`repeat`) |
| `Xla/Libs.lean` | the `Xla` DSL surface: `add`, `exp`, `einsum`, `gather`, `scatter`, `det`, `fori_loop`, `vmap`, `perm[...]`, ... |
| `Xla/Tensor.lean` | tensor library (`map`, `sumN`, `einsum`, `transpose`, `broadcast`, `flatten`, casts, ...) used by the semantics and proofs |
| `Xla/Impl.lean` | `DirectImpl`: the `Xla` semantics of every implemented primitive over `ℝ`/`ℤ` |
| `Xla/Smooth.lean`, `Xla/SmoothTactic.lean` | C² lemmas for the primitives, and the `contDiff_eval` tactic composing them |
| `Random/` | the probabilistic dialect: keys, distribution names, `RandVar`, `mean`, independence lemmas |
| `QMC/Ansatz.lean` | the validity contract `IsValidQMCWavefunction` and the `Ansatz` structure |
| `QMC/FermiNet.lean`, `QMC/PauliNet.lean` | SOTA-family molecular ansätze as single DSL programs, with validity proofs/design notes |
| `Examples/` | matmul, attention, normalize, Coulomb, sparse (COO) ops, `fori_loop`, permutation inverse, random-variable proofs |
| `python/` | JAX IR interpreter (`eval.py`, `ops.py`) and Lean→JAX bridge (`lean_eval.py`) |
| `docbuild/` | legacy doc-gen4 setup (pinned to an older toolchain) |

## The `Random` dialect

`Random` programs draw independent random numbers and combine them. A key is an `ℕ`
passed as an expression argument; `shuffle k = k + 1` derives a fresh key, so draws at
different keys are independent by construction. The semantics of a program is a
`RandVar` over the product law, and `mean` is its integral:

```lean
theorem mean_zsq (k : ℕ) : (zsq.eval k).mean = 1 := by
  simp [zsq, reduce_random, reduce_soir]
```

`Examples/random.lean` proves `E[Z] = 0`, `E[U] = 1/2`, `E[Z²] = 1`, `E[U²] = 1/3`,
independence of draws at distinct keys, and that an Euler–Maruyama noise step is
unbiased — i.e. Monte Carlo averages are *theorems*, not test thresholds.

## Worked example: verifiable QMC ansätze

`QMC/Ansatz.lean` collects the validity properties into `IsValidQMCWavefunction`, and
the design rule above says which tool each should be discharged with:

| Field | Failure it rules out | Verification |
|---|---|---|
| `antisymmetric` | bosonic collapse: a lower ground-state energy | numerical: evaluate the program on permuted inputs; a proof is optional |
| `smooth` (`ContDiff ℝ 2`) | creases/jumps: autodiff misses the distributional part of `ΔΨ`, biasing the energy invisibly | must be proved — no sampling test can detect it |
| `exp_decay` | blown-up or non-integrable densities that fake convergence before drifting/collapsing | numerical: drift/collapse monitors catch it, but only eventually; proved in the contract because the uniform structural certificate is cheaper than monitoring forever |

Smoothness is the field no test suite can replace. `Ansatz.lean` explains the design in
full, including why C² is the provable stand-in for `W^{2,1}_loc` and which failures
remain runtime-monitored.

`QMC/FermiNet.lean` implements the FermiNet architecture (shared-weight streams,
exponential envelopes, determinant heads, Jastrow factor) and proves the smoothness
field with `contDiff_eval`; `QMC/PauliNet.lean` implements a PauliNet/SchNet-style
ansatz with CRT-pooled orbital heads (fixed parameter count for all molecule sizes).
Their proof status is in the table below.

## Proof status

| Area | State |
|---|---|
| `Soir`, `Xla` core | complete: codegen, evaluator, tensor library, C² tactic |
| `Examples` | matmul, attention/softmax, normalize, Coulomb, `idxOfNonzero`, permutation inverse, `fori_loop`, transpose: fully proved; `denseToCoo`/`cooToDense`/`spmm` partially (`sorry`s) |
| `Random` | mean/independence lemmas proved; all `Examples/random.lean` theorems proved |
| `QMC.FermiNet` | `smooth` proved; `antisymmetric`, `exp_decay` are `sorry` |
| `QMC.PauliNet` | all three contract fields are `sorry` (documented as true by construction) |
| `DirectImpl` op coverage | analytic/sparse core implemented; primitives without a semantics case currently fall through to zero — only use the covered subset for executable programs |
| Python evaluator | mirrors `DirectImpl` by construction (verified numerically only through manual/informal checks in this repo, not a verified compiler) |

## Roadmap

* discharge the QMC `sorry`s (`antisymmetric`, `exp_decay`) and the sparse-example gaps;
* widen `DirectImpl` coverage (`log`, trig, `convert_type`, ...) and keep the Python
  evaluator in lockstep;
* use `Ansatz.isValid` as the fitness gate of an LLM-driven ansatz search — reject
  bosonic and non-smooth candidates before any Monte Carlo budget is spent.
