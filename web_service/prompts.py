"""LLM prompts for the SSA verification framework web service."""

FRAMEWORK_CONTEXT = """
You are working with a Lean 4 framework for verified XLA code generation.
The framework uses a custom SSA (Static Single Assignment) DSL to write programs,
and then proves theorems that the generated code computes exactly the mathematical specification.

Key files and patterns:
- Programs use `ssa Xla.XlaOp with ... begin ... end`
- Types are written as `⟨.float, [n, m]⟩` or `⟨.int, [n]⟩`
- `Tensor ℝ [n, m]` is `Fin n → Fin m → ℝ`
- `Tensor ℤ [n]` is `Fin n → ℤ`
- Programs are evaluated with `Xla.simpleEval program args...`

Xla DSL Operation Reference — EXACT signatures and usage patterns:

**Element-wise binary ops** (all take two `Expr XlaOp args [σ]` and return `Expr XlaOp args [σ]`):
- `Xla.add x y`, `Xla.sub x y`, `Xla.mul x y`, `Xla.div x y`, `Xla.mod x y`
- `Xla.div_int x y` — integer division (for int tensors)
- `Xla.and x y` — bitwise/logical AND (for int tensors)
- Example: `let z := Xla.mul x x`

**Element-wise unary ops (float tensors)**:
- `Xla.exp x`, `Xla.log x`, `Xla.sqrt x`, `Xla.cbrt x`
- `Xla.sin x`, `Xla.cos x`, `Xla.cosh x`
- `Xla.acos x`, `Xla.asin x`, `Xla.atan x`
- `Xla.acosh x`, `Xla.asinh x`, `Xla.atanh x`
- `Xla.erf x`, `Xla.erfc x`, `Xla.expm1 x`
- Example: `let z := Xla.log x`

**Element-wise unary ops (any dtype)**:
- `Xla.neg x`, `Xla.abs x`, `Xla.exp2 x`
- Example: `let z := Xla.abs x`

**Type-conversion ops**:
- `Xla.ceil x` — float → int (e.g., `⟨.float, [n]⟩` → `⟨.int, [n]⟩`)
- `Xla.convert_type (α := .float) (β := .int) x` — general dtype cast
- Example: `let i := Xla.ceil x; let f := Xla.convert_type (α := .int) (β := .float) i`

**Xla.einsum** — the most general operation; `transpose`, `dot_general`, and `sum` can all be expressed as special cases of `einsum`:
- Signature: `Xla.einsum (s : Shape) (indices : List (List (Fin s.length))) (n : ℕ) (xs : Expr XlaOp args (...))`
- `s` is the full shape of the contraction space.
- `indices` maps each input tensor to its dimensions in `s`.
- `n` is the number of leading dimensions in `s` that are summed over (contracted). The output shape is `s.drop n`.
- Multiple inputs must be appended: `Xla.einsum s indices n (x.append y)`
- **Examples:**
  - Sum all elements of vector `x : [N]`: `Xla.einsum [N] [[0]] 1 x`
  - Sum over first axis of matrix `A : [N, M]`: `Xla.einsum [N, M] [[0, 1]] 1 A`
  - Transpose `A : [N, M]` to `[M, N]`: `Xla.einsum [M, N] [[1, 0]] 0 A`
  - Matrix-vector: `A : [N, M]`, `v : [M]` → `Xla.einsum [M, N] [[1, 0], [0]] 1 (A.append v)`
  - Matrix multiply: `A : [N, M]`, `B : [M, L]` → `Xla.einsum [M, N, L] [[1, 0], [0, 2]] 1 (A.append B)`
  - Batch matmul (from SchNet): `Xla.einsum [n_base, n_atom, n_atom, n_filter] [[1,2,0], [0,3]] 1 (rbf.append filter)`

**Xla.sum** — sum over the FIRST `n` dimensions (shorthand for einsum):
- `Xla.sum (n : ℕ) (x : Expr XlaOp args [⟨α, s⟩]) : Expr XlaOp args [⟨α, s.drop n⟩]`
- For a vector `x : ⟨.float, [N]⟩`, use `Xla.sum 1 x` to sum all elements into a scalar.
- For a matrix `x : ⟨.float, [N, M]⟩`, `Xla.sum 1 x` sums over rows (produces `[M]`), `Xla.sum 2 x` sums over both dims (produces `[]`).

**Xla.broadcast** — EXACT signature:
- `Xla.broadcast (dims : List (ℕ × Bool)) (x : Expr XlaOp args [⟨α, Tensor.preBroadcast dims⟩]) : Expr XlaOp args [⟨α, dims.map Prod.fst⟩]`
- `dims` is a list of `(size, bool)` pairs. `true` means the original dimension is preserved in that position; `false` means a new dimension is inserted.
- Example: broadcast a scalar `x : ⟨.float, []⟩` to a vector of length `n`:
  `Xla.broadcast [⟨n, false⟩] x`
- Example: broadcast a vector `x : ⟨.float, [n]⟩` to matrix `[n, m]` by replicating along the second axis:
  `Xla.broadcast [⟨n, true⟩, ⟨m, false⟩] x`
- Example (from Coulomb): broadcast `x : ⟨.float, [n_atom, 3]⟩` to shape `[n_atom, n_atom, 3]`:
  `Xla.broadcast [⟨n_atom, true⟩, ⟨n_atom, false⟩, ⟨3, true⟩] x`

**Xla.transpose / Xla.transpose'** (can be replaced by einsum):
- `Xla.transpose perm x` where `perm : Equiv.Perm (Fin s.length)`
- `Xla.transpose' x σ` where `σ : Fin s.length → Fin s.length` (auto-proves injective/surjective)
- Example: `Xla.transpose [0,1].formPerm A` for a matrix (no-op transpose)
- Example: `Xla.transpose' x (![2,0,1] : Fin 3 → Fin 3)` for a 3D tensor

**Xla.dot_general** (can be replaced by einsum):
- `Xla.dot_general batch contract lhs rhs x y`
- `x` must have shape `contract ++ batch ++ lhs`
- `y` must have shape `contract ++ batch ++ rhs`
- Output shape is `batch ++ lhs ++ rhs`
- Example (matrix-vector): `Xla.dot_general [] [m] [n] [] A_t v` where `A_t` is `[m, n]` and `v` is `[m]`, output is `[n]`.
- Example (matmul): `Xla.dot_general [] [m] [n] [l] A_t B` where `A_t` is `[m, n]` and `B` is `[m, l]`, output is `[n, l]`.

**Xla.gather**:
- `Xla.gather (x : Expr XlaOp args [⟨α, s⟩]) (indices : Expr XlaOp args (List.replicate s.length ⟨.int, s'⟩))`
- Output has shape `s'`.
- Example: `Xla.gather θ (x.append i)` where `x : ⟨.int, []⟩` and `i : ⟨.int, [embed_dim]⟩`.

**Xla.scatter**:
- `Xla.scatter (x : Expr XlaOp args [⟨α, s⟩]) (y : Expr XlaOp args [⟨α, [n]⟩]) : Curry (fun ι ↦ Expr XlaOp args [ι]) (List.replicate s.length ⟨.int, [n]⟩) (Expr XlaOp args [⟨α, s⟩])`
- Takes base tensor and updates vector, then returns a curried function expecting index tensors.
- Example: `Xla.scatter (s := [n]) (Xla.const_float [n] 0) (Xla.iota n) x`

**Xla.choice** (where/conditional):
- `Xla.choice (c : Expr XlaOp args [⟨.int, s⟩]) (x y : Expr XlaOp args [⟨α, s⟩])`
- Returns `x` where `c != 0`, else `y`.
- Example: `Xla.choice mask (Xla.const_float [n, n] 0) d`

**Xla.cumsum**:
- `Xla.cumsum x` — cumulative sum along the first axis.

**Xla.iota**:
- `Xla.iota (n : ℕ)` — generates `[0, 1, 2, ..., n-1]` with type `⟨.int, [n]⟩`.

**Xla.const_float / Xla.const_int**:
- `Xla.const_float (s : Shape) (val : ℕ)` — creates a float tensor of shape `s` filled with `val`.
- `Xla.const_int (s : Shape) (val : ℕ)` — creates an int tensor of shape `s` filled with `val`.
- Example: `Xla.const_float [n, m] 0` — zero-filled `n × m` float matrix.
- Example: `Xla.const_int [n] 1` — vector of int 1s.
- Example: `Xla.const_float [] 2` — scalar float `2.0`.

**Xla.fori_loop**:
- `Xla.fori_loop (body : Expr XlaOp (carry ++ aux) carry) (n : Expr XlaOp args [⟨.int, []⟩]) (init : Expr XlaOp args carry) (aux_val : Expr XlaOp args aux)`
- Example: `Xla.fori_loop power_loop_body m x .nil`

**Xla.vmap**:
- `Xla.vmap (batch : ℕ) (fn : Expr XlaOp (ins ++ aux) outs) (xs : Expr XlaOp args (ins.map fun ⟨α, s⟩ ↦ ⟨α, batch :: s⟩)) (aux : Expr XlaOp args aux)`
- Example: `Xla.vmap (ins := [⟨.float, [3]⟩]) n_atom periodic_distance x lattice`

**Xla.eq**:
- `Xla.eq x y` — returns `⟨.int, s⟩` with 1 where equal, 0 where not equal.

**Reduction ops**:
- `Xla.argmax axis x` — returns index of maximum along `axis` as `⟨.int, ...⟩`
- `Xla.argmin axis x` — returns index of minimum along `axis` as `⟨.int, ...⟩`
- Example: `let_expr idx : [⟨.int, [m]⟩] := Xla.argmax 0 A` where `A : [n, m]`

**Cumulative ops**:
- `Xla.cumsum x` — cumulative sum along the first axis
- `Xla.cumlogsumexp axis reverse x` — cumulative log-sum-exp
- `Xla.cummax axis reverse x`, `Xla.cummin axis reverse x`, `Xla.cumprod axis reverse x`
- `reverse` defaults to `false`
- Example: `let y := Xla.cummax 0 false x`

**Tensor manipulation**:
- `Xla.concat batch axis x y` — concatenate `x` and `y` along `axis`
  - Example (concatenate two vectors): `Xla.concat [] 0 x y` where `x : [n]`, `y : [m]`
  - Example (concatenate along batch dim): `Xla.concat [b] 1 x y` where `x : [b, n]`, `y : [b, m]`
- `Xla.sorted batch x` — sort along the last axis
  - Example: `Xla.sorted [] x` where `x : [n]`
- `Xla.empty` — create an empty tensor (may need `let_expr` for type inference)

**Linear algebra**:
- `Xla.cholesky batch A` — Cholesky decomposition
  - Example: `Xla.cholesky [] A` where `A : [n, n]`
- `Xla.eigvals batch A` — eigenvalues (returns `⟨.float, batch ++ [n]⟩`)
- `Xla.eigvalsh batch A` — eigenvalues of Hermitian matrix
- `Xla.eigvecs batch A` — eigenvectors (returns `⟨.float, batch ++ [n, n]⟩`)
- `Xla.eigvecsh batch A` — eigenvectors of Hermitian matrix
  - Example: `let_expr v : [⟨.float, [n]⟩] := Xla.eigvals [] A`

**Binding expressions**:
- Use `let_expr name : [type] := expr;` to bind SSA expressions. The `let_expr` syntax is REQUIRED when Lean cannot infer the implicit `args` parameter (e.g., for `Xla.iota`, `Xla.const_float`/`const_int` with no other context, `Xla.einsum`, `Xla.transpose`, `Xla.dot_general`, sub-program `.apply`, etc.).
- `let name := expr;` may work when `args` can be inferred from surrounding expressions (e.g., `Xla.add x y` where `x` and `y` are already bound).

**Composing expressions**:
- `x.append y` — concatenate two multi-output expressions into one multi-output expression.
- `fn.apply args` — call a previously-defined sub-program. `args` must be a multi-output expression matching the sub-program's input types.
- `.nil` — empty expression (used for empty auxiliary arguments, e.g., in `fori_loop` and `vmap`).
"""

FORMALIZATION_CONTEXT = """
You are working with a Lean 4 framework for verified XLA code generation.
Your task is to write a mathematical specification (as a pure Lean function) and a theorem statement asserting that an SSA program computes exactly that specification.

Key concepts:
- `SSA.Tensor ℝ [n]` represents a vector of length `n` of real numbers (i.e., `Fin n → ℝ`).
- `SSA.Tensor ℝ [n, m]` represents an `n × m` matrix (i.e., `Fin n → Fin m → ℝ`).
- `SSA.Tensor ℤ [n]` represents a vector of length `n` of integers.
- ALWAYS use `Xla.simpleEval program args...` in theorem statements.
- NEVER use `.eval Xla.DirectImpl` or `Index.single` in theorem statements.
- `Xla.simpleEval` is for single-output expressions and directly returns the tensor result.

Theorem statement patterns:
- `Xla.simpleEval myProgram x y = my_spec x y`
- Use `example` or `theorem` with explicit natural number parameters.
- Write the mathematical definition as a `def` with a `_spec` suffix.
- Use `SSA.Tensor ℝ [...]` for all tensor-typed arguments, even 1D vectors.
- If the spec involves noncomputable real operations (e.g. `√`, `Real.exp`), mark it as `noncomputable def`.
"""

FORMALIZE_PROMPT = """
You are given a natural language specification for a scientific computing function.
Your task is to produce a **mathematical formalization** in Lean 4 for the SSA framework.

Output exactly the following (no extra prose):
1. A mathematical definition of what the function computes (as a pure Lean function)
2. A theorem statement asserting that `Xla.simpleEval` of the program equals this mathematical definition

Rules:
- Do NOT write the SSA program implementation yet.
- Do NOT write the proof.
- ALWAYS use `Xla.simpleEval program args...` in the theorem statement.
- NEVER use `.eval Xla.DirectImpl` or `Index.single`.
- Use `SSA.Tensor ℝ [n]` for vectors, `SSA.Tensor ℝ [n, m]` for matrices, `SSA.Tensor ℤ [n]` for integer vectors.
- Include proper type signatures with natural number parameters.
- AVOID common Mathlib names like `normalize`, `map`, `sum`, `prod`, `filter`, `range`, `compose`, etc.
  Use descriptive names (e.g., `l2_normalize_spec`, `vector_sum_spec`, `matmul_spec`) to prevent conflicts.

Example 1 (Vector L2 norm):
```lean
import SSA

noncomputable def norm_spec {n : ℕ} (x : SSA.Tensor ℝ [n]) : ℝ :=
  √(∑ i, (x i) ^ 2)

theorem norm_correct (n : ℕ) (x : SSA.Tensor ℝ [n]) :
    Xla.simpleEval norm_xla x = norm_spec x := by
  sorry
```

Example 2 (Matrix multiplication):
```lean
import SSA

theorem matmul_correct (n m l : ℕ) (x : SSA.Tensor ℝ [n, m]) (y : SSA.Tensor ℝ [m, l]) :
    Xla.simpleEval matmul x y = fun i j => ∑ k, x i k * y k j := by
  sorry
```

Example 3 (Coulomb matrix):
```lean
import SSA

open SSA in
example {n_atom : ℕ} (x : SSA.Tensor ℝ [n_atom, 3]) :
    Xla.simpleEval Coulomb x = ∑ i, ∑ j with i ≠ j, 1 / √(∑ k, (x i k - x j k) ^ 2) := by
  sorry
```

Example 4 (Vector dot product):
```lean
import SSA

def dot_spec {n : ℕ} (x y : SSA.Tensor ℝ [n]) : ℝ :=
  ∑ i, x i * y i

theorem dot_correct (n : ℕ) (x y : SSA.Tensor ℝ [n]) :
    Xla.simpleEval dot x y = dot_spec x y := by
  sorry
```

Example 5 (Element-wise vector addition):
```lean
import SSA

def add_spec {n : ℕ} (x y : SSA.Tensor ℝ [n]) : SSA.Tensor ℝ [n] :=
  fun i => x i + y i

theorem add_correct (n : ℕ) (x y : SSA.Tensor ℝ [n]) :
    Xla.simpleEval vadd x y = add_spec x y := by
  sorry
```

Example 6 (Sum of all elements in a matrix):
```lean
import SSA

def sum_all_spec {n m : ℕ} (A : SSA.Tensor ℝ [n, m]) : ℝ :=
  ∑ i, ∑ j, A i j

theorem sum_all_correct (n m : ℕ) (A : SSA.Tensor ℝ [n, m]) :
    Xla.simpleEval sum_all A = sum_all_spec A := by
  sorry
```

Example 7 (Cumulative sum of a vector):
```lean
import SSA

def cumsum_spec {n : ℕ} (x : SSA.Tensor ℝ [n]) : SSA.Tensor ℝ [n] :=
  fun i => ∑ j with j ≤ i, x j

theorem cumsum_correct (n : ℕ) (x : SSA.Tensor ℝ [n]) :
    Xla.simpleEval cumsum x = cumsum_spec x := by
  sorry
```

Example 8 (Periodic distance):
```lean
import SSA

open SSA in
example (x : SSA.Tensor ℝ [3]) (lattice : ℝ) :
    Xla.simpleEval periodic_distance x lattice =
    ∑ i, ((x i + lattice / 2) % lattice - lattice / 2) ^ 2 := by
  sorry
```

Example 9 (Permutation inverse with integer output):
```lean
import SSA

theorem permInv_correct (n : ℕ) (hn : n ≠ 0) (σ : Equiv.Perm (Fin n)) :
    Xla.simpleEval permInv (fun i => σ i) = fun i => (σ.symm i : ℤ) := by
  sorry
```

Example 10 (Message passing — tensor output):
```lean
import SSA

def message_passing_spec {n_atom n_filter n_feat : ℕ}
  (x : SSA.Tensor ℝ [n_atom, n_feat])
  (rbf : SSA.Tensor ℝ [n_atom, n_atom, n_filter])
  (proj₀ proj₁ : SSA.Tensor ℝ [n_feat, n_filter]) :
    SSA.Tensor ℝ [n_atom, n_feat] :=
  fun atom feat =>
    ∑ filter,
      (∑ atom', rbf atom' atom filter * (∑ feat', x atom' feat' * proj₀ feat' filter))
      * proj₁ feat filter

theorem message_passing_correct {n_atom n_filter n_feat : ℕ}
  (x : SSA.Tensor ℝ [n_atom, n_feat])
  (rbf : SSA.Tensor ℝ [n_atom, n_atom, n_filter])
  (proj₀ proj₁ : SSA.Tensor ℝ [n_feat, n_filter]) :
    Xla.simpleEval message_passing x rbf proj₀ proj₁ = message_passing_spec x rbf proj₀ proj₁ := by
  sorry
```

Now formalize this specification:

{spec}
"""

IMPLEMENT_PROMPT = """
You are given a Lean 4 theorem statement for the SSA verification framework.
Your task is to:
1. Write the SSA program implementation using the `ssa` DSL
2. Add the theorem statement with `sorry` as the proof body
3. Add an `#eval IO.println (program (n:=4) (m:=4) ...).code` line to generate IR

Do NOT write any proof tactics. Use `sorry` for all proof bodies.
Output exactly a complete, compilable Lean 4 file. Do NOT output markdown code blocks — output raw Lean code only.

Here are examples of complete implementations:

Example 1 (Vector element-wise square):
```lean
import SSA

def square {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    return Xla.mul x x

#eval IO.println (square (n:=8)).code

example (n : ℕ) (x : SSA.Tensor ℝ [n]) :
    Xla.simpleEval square x = fun i => (x i) ^ 2 := by
  sorry
```

Example 2 (Sum of elements):
```lean
import SSA

def sum_all {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let_expr z : [⟨.float, []⟩] := Xla.einsum [n] [[0]] 1 x;
    return z

#eval IO.println (sum_all (n:=8)).code

example (n : ℕ) (x : SSA.Tensor ℝ [n]) :
    Xla.simpleEval sum_all x = ∑ i, x i := by
  sorry
```

Example 3 (Matrix-vector product):
```lean
import SSA

def matvec {n m : ℕ} :=
  ssa Xla.XlaOp with
    A : ⟨.float, [n, m]⟩,
    v : ⟨.float, [m]⟩
  begin
    let_expr z : [⟨.float, [n]⟩] := Xla.einsum [m, n] [[1, 0], [0]] 1 (A.append v);
    return z

#eval IO.println (matvec (n:=4) (m:=4)).code

example (n m : ℕ) (A : SSA.Tensor ℝ [n, m]) (v : SSA.Tensor ℝ [m]) :
    Xla.simpleEval matvec A v = fun i => ∑ j, A i j * v j := by
  sorry
```

Example 4 (Vector normalization with broadcast):
```lean
import SSA

def l2_norm {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let x2  := Xla.mul x x;
    let_expr x2_sumed  : [⟨.float, []⟩] := Xla.einsum [n] [[0]] 1 x2;
    return Xla.sqrt x2_sumed

def l2_normalize {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let x_norm := l2_norm.apply x;
    let x_norm_broadcasted := Xla.broadcast [⟨n, false⟩] x_norm;
    return Xla.div x x_norm_broadcasted

#eval IO.println (l2_normalize (n := 12)).code

noncomputable def l2_normalize_spec {n : ℕ} (x : SSA.Tensor ℝ [n]) : SSA.Tensor ℝ [n] :=
  fun i => x i / √(∑ j, (x j)^2)

example (n : ℕ) (x : SSA.Tensor ℝ [n]) :
    Xla.simpleEval l2_normalize x = l2_normalize_spec x := by
  sorry
```

Example 5 (Zero-mean vector — sum, broadcast, sub):
```lean
import SSA

def zero_mean {n : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n]⟩
  begin
    let_expr sum_n : [⟨.float, []⟩] := Xla.einsum [n] [[0]] 1 x;
    let mean := Xla.div sum_n (Xla.const_float [] n);
    let mean_broadcasted := Xla.broadcast [⟨n, false⟩] mean;
    return Xla.sub x mean_broadcasted

#eval IO.println (zero_mean (n:=8)).code

example (n : ℕ) (x : SSA.Tensor ℝ [n]) :
    Xla.simpleEval zero_mean x = fun i => x i - (∑ j, x j) / n := by
  sorry
```

Example 6 (Mutual distance between two groups of 3D points — broadcast, sub, mul, einsum, sqrt):
```lean
import SSA

def mutual_distance {n m : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n, 3]⟩,
    y : ⟨.float, [m, 3]⟩
  begin
    let x_broadcast := Xla.broadcast [⟨n, true⟩, ⟨m, false⟩, ⟨3, true⟩] x;
    let y_broadcast := Xla.broadcast [⟨n, false⟩, ⟨m, true⟩, ⟨3, true⟩] y;
    let diff := Xla.sub x_broadcast y_broadcast;
    let diff2 := Xla.mul diff diff;
    let_expr dist2 : [⟨.float, [n, m]⟩] := Xla.einsum [3, n, m] [[1, 2, 0]] 1 diff2;
    return Xla.sqrt dist2

#eval IO.println (mutual_distance (n:=4) (m:=5)).code

example (n m : ℕ) (x : SSA.Tensor ℝ [n, 3]) (y : SSA.Tensor ℝ [m, 3]) :
    Xla.simpleEval mutual_distance x y = fun i j => √(∑ k, (x i k - y j k) ^ 2) := by
  sorry
```

Now implement the following theorem. Use default small sizes (like 4 or 8) for the #eval.
Use `sorry` for all proof bodies. Do NOT write any proof tactics.

IMPORTANT: AVOID common Mathlib names like `normalize`, `map`, `sum`, `prod`, `filter`, `range`, `compose`, etc.
Use descriptive, specific names (e.g., `l2_normalize`, `vector_sum`, `matmul`) to prevent name conflicts.
The program name must match what the theorem statement expects (e.g. if the theorem uses `l2_normalize`, define `def l2_normalize`).

{theorem_text}
"""

FIX_PROMPT = """
The following Lean 4 file failed to compile. Here is the error:

--- ERROR ---
{error}
--- END ERROR ---

Here is the current file content:

--- FILE ---
{lean_code}
--- END FILE ---

Please fix the compilation error and output the complete corrected file.
Do NOT output markdown code blocks — output raw Lean code only.
Do NOT change any proof bodies — leave them as `sorry`.
Common fixes:
- Missing imports
- Type mismatches in SSA expressions
- Wrong number of arguments to Xla ops
- Incorrect broadcast dimensions
- Wrong shape in `dot_general`, `einsum`, `transpose`, or `gather`
- Use `Xla.const_float shape val` or `Xla.const_int shape val` for constant tensors (may need `let_expr`)
- Use `Xla.iota n` (not `Xla.iota`)
- Use `let_expr name : [type] := expr;` when `args` cannot be inferred
- Use `Xla.einsum` instead of `Xla.transpose`, `Xla.dot_general`, or `Xla.sum` where applicable
"""
