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
cheapest one that actually works for it. We use [QMC](./QMC) as an example. Suppose we want to use
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

## Example

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
