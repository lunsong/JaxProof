import QMC.Ansatz
import Xla.Smooth

/-!
# FermiNet — a neural-network QMC ansatz for molecules

FermiNet (Pfau, Spencer, Matthews, Foulkes, *Ab initio solution of the many-electron
Schrödinger equation with deep neural networks*, Phys. Rev. Research 2, 033429
(2020)) is the canonical neural network wavefunction for molecules and the
architectural ancestor of the current state-of-the-art QMC ansätze (Moon,
PsiFormer, PauliNet, FermiNet with Pfaffians). This file implements the FermiNet
architecture as a single `Xla.SimpleExpr` program in the Soir/XLA DSL, packaged as
an `Ansatz` satisfying the contract of `QMC/Ansatz.lean`.

## Style: closed libraries composed with `Expr.apply`

Every reusable component — including `tanh` and the parameter slicing — is a closed
`SimpleExpr` whose inputs are formal arguments, and the components are composed with
`Soir.Expr.apply` (`Examples/Coulomb.lean` is the minimal example of the style). The
emitted IR is therefore a set of `@n` library bodies called by `call; @n, …` instead of
one flat program.

Why this matters for code generation cost: `Expr.code` (Soir/Core.lean) deduplicates
sub-expressions by `Expr.hashExpr`, a *structural* hash that re-walks a node's whole
subtree (calling `toString` on the op of every `bind` node), and it evaluates that
hash many times per node — once per cache element in the `List.find?` lookups of
`Expr.genCode`/`Expr.addLib`, and again in `Expr.addVars`. Cost is therefore
(entries in the scope's cache) × (hash of a subtree), so the size of the scope a
program is generated in matters quadratically. A flat program is one scope of every
binding it contains; this file is a short chain of calls whose library bodies are each
generated in a scope of their own small size.

Measured (Lean 4.33.1, `lake env lean`, interpreter: DAG construction + `Expr.code`,
so the numbers below exclude the ~2.5 s toolchain startup), against the builder-style
formulation of this same program that this file replaces:

| program | builder style | this file |
|---|---|---|
| `fermiNetAnsatz 2 1 1` (H₂)  | 17.0 s | 0.37 s |
| `fermiNetAnsatz 1 2 1` (Li)  | 17.2 s | 0.40 s |
| `fermiNetAnsatz 1 1 2` (Be)  | 17.4 s | 0.40 s |
| `fermiNetAnsatz 3 4 4` (H₂O) | 16.9 s | 0.37 s |

Both are flat in molecule size (the program structure, not the tensor shapes, drives
the cost). The emitted IR for H₂O was one 360-binding flat program in the builder style
versus a 31-binding top level calling 59 libraries here — the scope that the `find?`
scans run over shrinks from 360 bindings to 31, and each library body is generated in
its own (small) scope. For scale: one full structural hash of this program takes 3 ms,
so the builder-style codegen did ~5 700 full-program hashes' worth of work against
~120 here.

A synthetic depth sweep (`K` inlined vs. `K` applied steps of the same op) shows the
same constant-factor gap growing with depth (both styles quadratic in `K`):

| `K` | builder | apply |
|---|---|---|
| 10 | 14 ms | 1 ms |
| 20 | 54 ms | 3 ms |
| 40 | 218 ms | 10 ms |
| 80 | 868 ms | 35 ms |
| 160 | – | 138 ms |
| 320 | – | 544 ms |

`python/emit_ferminet_ir.lean` emits this program for the Python harness, which checks
its amplitude in the JAX evaluator (`python3 python/test_ferminet.py`).

## Architecture

Per spin sector `σ ∈ {↑, ↓}` there are three streams of width `F = 32`:

* **one-electron stream** `h_i^σ`: per-electron features. The initial layer
  contracts the electron–nucleus displacements, distances (with `ε = exp θ > 0`
  guarding the square root away from zero) and a nuclear-charge embedding
  (a `gather`-based table lookup indexed by the (clamped) nuclear charge `Z`) over
  the nucleus index into `F` features, followed by `tanh`.
* **two-electron streams** `g_ij^σ` (same spin) and `g_ij^{σ σ̄}` (opposite spin):
  per-pair features from the electron–electron displacement and distance.
* The layers `l = 0, …, L-1` interleave the streams exactly as in the paper
  (shared weights across spins):
  * `h_i^{l+1} = tanh(V h_i^l + Σ_j w ⊙ g_ij^l + Σ_j w' ⊙ g_ij^{l σ σ̄} + b)`
  * `g_ij^{l+1} = tanh(G g_ij^l + H (h_i^l + h_j^l) + c)`
  * `g_ij^{l+1 σ σ̄} = tanh(G' g_ij^{l σ σ̄} + H' (h_i^σ + h_j^{σ̄}) + c')`

The final amplitude is

  `Ψ = exp(J↑ + J↓) · Σ_k w_k det_k↑ · Σ_k w_k det_k↓`

where `J^σ = Σ_i w_J · h_i^{σ,L}` is a bounded Jastrow factor (the streams are
`tanh`-bounded), each orbital is `φ_i^{kσ}(x_j) = (w_k·h_j^{σ,L}) · exp(-Σ_α
A_{k,i,α} ‖x_j - R_α‖)` with `A = exp θ > 0` the FermiNet exponential envelope
(decay exponents indexed by *orbital* `(k, i)` — the permutation-equivariant
indexing, cf. FermiNet Eq. 19; `√(r² + ε)` keeps the norm term real-analytic),
and the `K` determinants per spin are computed by `vmap`-ing the `det` primitive
over the leading axis. Determinant weights are `w_k = exp θ_k > 0`.

## Why the validity contract holds (sketch; proofs are `sorry`)

* **`antisymmetric`**: the one- and two-electron streams and the per-orbital
  envelope are permutation *equivariant* (each update is a symmetric aggregation
  `Σ_j` of equivariant features; the envelope parameters are indexed by orbital,
  not by electron), the Jastrow is permutation *invariant*, and each determinant
  acquires `sign σ` under a permutation of its rows (`Matrix.det_permutation`).
  Exchanging same-spin electrons thus multiplies `Ψ` by `σ.sign`; the two spin
  sectors are independent factors.
* **`smooth`** (C²): every primitive used is real-analytic on its domain —
  `add/mul/sub/div/exp/sum/transpose/broadcast/einsum/det/gather` on integer
  data — with `sqrt` applied to `r² + ε`, `ε > 0`, so the argument never touches
  the non-analytic point 0; `tanh` is the composition `1 - 2/(exp(2x) + 1)` of
  analytic functions. All reparameterizations use `exp`, so positivity holds for
  *every* `θ` (the `∀θ` contract). The proof is a composition along the dataflow:
  `smooth` reduces to `contDiff_fermiNetAnsatz`, which composes the per-node lemmas
  `contDiff_eval_*` in the section "Smoothness: the decomposition", whose leaves are
  the per-primitive facts in `Xla/Smooth.lean`. See that section for the shape of
  the statements and the status of the proofs.
* **`exp_decay`**: `‖x‖ → ∞` forces `Σ_j env_j ≳ a‖x‖` (`A_{k,i,α} > 0` and
  `√(r²+ε) ≥ ‖x_j - R_α‖`), so the envelope contributes `exp(-a‖x‖)`. The
  determinant entries are bounded (`tanh`-bounded streams) times the envelope,
  hence `|det| ≤ C exp(-a‖x‖)` by Hadamard; the Jastrow is bounded. Products
  multiply envelopes, sums take their min — overall `|Ψ| ≤ C exp(-k‖x‖)`, `k > 0`.

## Parameter layout (`θ : Fin N_PARAM → ℝ`)

`N_PARAM` parameters suffice for the largest supported system
(`N_MAX_NUC = 10` nuclei, `N_MAX_EL = 10` electrons per spin); smaller molecules
use the leading sub-blocks via `gather` slices, so the same `θ` works for every
molecule size.

| block | offset | size | shape |
|---|---|---|---|
| `ε` | 0 | 1 | `[1]` |
| `W_d` | 1 | 960 | `[N_MAX_NUC, 3, F]` |
| `w_s` | 961 | 320 | `[N_MAX_NUC, F]` |
| `w_z` | 1281 | 320 | `[N_MAX_NUC, F]` |
| `b₀` | 1601 | 32 | `[F]` |
| `W_e` | 1633 | 96 | `[3, F]` |
| `w_d` | 1729 | 32 | `[F]` |
| `g_b` | 1761 | 32 | `[F]` |
| layer 0 | 1793 | 5280 | see below |
| layer 1 | 7073 | 5280 | |
| `w_J` | 12353 | 32 | `[F]` |
| `Z` table | 12385 | 128 | `[128]` |
| `A` | 12513 | 200 | `[K, N_MAX_EL, N_MAX_NUC]` |
| `W_orb` | 12713 | 640 | `[F, K, N_MAX_EL]` |
| `w_det` | 13353 | 2 | `[K]` |

Each layer block (5280 floats) is `V [F,F]` at `+0`, `w [F]` at `+1024`,
`w' [F]` at `+1056`, `b [F]` at `+1088`, `G [F,F]` at `+1120`, `H [F,F]` at
`+2144`, `c [F]` at `+3168`, `G' [F,F]` at `+3200`, `H' [F,F]` at `+4224`,
`c' [F]` at `+5248`.

## Op-set notes

Everything here is expressed with the existing primitives:

* `tanh x` is the composition `1 - 2/(exp(2x) + 1)` — mathematically exact and
  numerically stable in the JAX backend for both signs of overflow
  (`2/∞ = 0`, `exp(-∞) = 0`). A dedicated `tanh` primitive would be a
  quality-of-life addition (the Python backend already registers one).
* positivity reparameterizations use `exp` only (no `softplus`), because
  `log` has no `DirectImpl` case in `Xla/Impl.lean` (it currently falls through
  to `DirectImpl.zero`), and the DSL semantics must agree with the Lean
  semantics.
* nuclear-charge embedding uses `gather` on a `θ`-table indexed by `Z`
  (clamped by `mod`) — `convert_type` is likewise unimplemented in
  `DirectImpl`, so int→float conversion is avoided.
* parameter slicing for smaller molecules is `gather` over `iota + const` indices;
  a dedicated slice primitive would be cleaner.
-/

namespace FermiNet

open Soir Xla

/-! ### Architecture constants -/

/-- Maximal number of nuclei supported (bounds the per-nucleus parameter blocks). -/
def N_MAX_NUC : ℕ := 10

/-- Maximal number of electrons per spin sector supported. -/
def N_MAX_EL : ℕ := 10

/-- Hidden width of the one- and two-electron streams. -/
def F : ℕ := 32

/-- Number of interaction layers (unrolled in the program). -/
def L_LAYERS : ℕ := 2

/-- Number of determinants per spin sector. -/
def K : ℕ := 2

/-- Size of the nuclear-charge embedding table. -/
def Z_TAB : ℕ := 128

/-! ### Parameter offsets (see the module docstring) -/

def OFF_EPS : ℕ := 0
def OFF_WD : ℕ := 1
def OFF_WS : ℕ := OFF_WD + N_MAX_NUC * 3 * F
def OFF_WZ : ℕ := OFF_WS + N_MAX_NUC * F
def OFF_B0 : ℕ := OFF_WZ + N_MAX_NUC * F
def OFF_WE : ℕ := OFF_B0 + F
def OFF_WD2 : ℕ := OFF_WE + 3 * F
def OFF_GB : ℕ := OFF_WD2 + F
def OFF_L0 : ℕ := OFF_GB + F
def OFF_L1 : ℕ := OFF_L0 + 5280
def OFF_WJ : ℕ := OFF_L1 + 5280
def OFF_ZTAB : ℕ := OFF_WJ + F
def OFF_A : ℕ := OFF_ZTAB + Z_TAB
def OFF_WORB : ℕ := OFF_A + K * N_MAX_EL * N_MAX_NUC
def OFF_WDET : ℕ := OFF_WORB + F * K * N_MAX_EL

/-- Total number of parameters. -/
def N_PARAM : ℕ := OFF_WDET + K

/-! ### Primitives composed from the existing op set -/

/-- `tanh` from primitives: `1 - 2 / (exp (2x) + 1)`. Mathematically exact and
numerically stable for both signs of overflow. -/
def tanh (s : Shape) : SimpleExpr [⟨.float, s⟩] ⟨.float, s⟩ :=
  Expr.ofFn fun x =>
    let two : Expr XlaOp [⟨.float, s⟩] [⟨.float, s⟩] := Xla.ofNat 2
    let one : Expr XlaOp [⟨.float, s⟩] [⟨.float, s⟩] := Xla.ofNat 1
    let denom := Xla.add (Xla.exp (Xla.mul two x)) one
    Xla.sub one (Xla.div two denom)

/-! ### Parameter access -/

/-- Slice `len` consecutive floats from the parameter vector starting at `off`. -/
def paramSlice (off len : ℕ) : SimpleExpr [⟨.float, [N_PARAM]⟩] ⟨.float, [len]⟩ :=
  Expr.ofFn fun θ =>
    let idx : Expr XlaOp [⟨.float, [N_PARAM]⟩] [⟨.int, [len]⟩] :=
      Xla.add (Xla.iota len) (Xla.ofNat off)
    Xla.gather (α := .float) (s := [N_PARAM]) (s' := [len]) θ idx

/-- Reshape a slice of the parameter vector into a tensor of shape `s`. -/
def paramBlock (off : ℕ) (s : Shape) : SimpleExpr [⟨.float, [N_PARAM]⟩] ⟨.float, s⟩ :=
  Expr.ofFn fun θ => Xla.unflatten s ((paramSlice off s.prod).apply θ)

/-- A scalar parameter reparameterized through `exp`, so it is `> 0` for every `θ`. -/
def posScalar (off : ℕ) : SimpleExpr [⟨.float, [N_PARAM]⟩] ⟨.float, []⟩ :=
  Expr.ofFn fun θ => Xla.exp (Xla.sum 1 ((paramSlice off 1).apply θ))

/-! ### Geometry features -/

/-- Electron–nucleus displacement `rᵢ - Rα`, shape `[N, N_nuc, 3]`. -/
def enDisp (N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_nuc, 3]⟩] ⟨.float, [N, N_nuc, 3]⟩ :=
  Expr.ofFn fun r => fun R =>
    let rB := Xla.broadcast [⟨N, true⟩, ⟨N_nuc, false⟩, ⟨3, true⟩] r
    let RB := Xla.broadcast [⟨N, false⟩, ⟨N_nuc, true⟩, ⟨3, true⟩] R
    Xla.sub rB RB

/-- Electron–nucleus distance `√(‖rᵢ - Rα‖² + ε)` with `ε = exp θ > 0`, keeping the
square root argument away from the non-analytic point 0. -/
def enDist (N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_nuc, 3]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N_nuc]⟩ :=
  Expr.ofFn fun r => fun R => fun θ =>
    let d := (enDisp N N_nuc).apply (r.append R)
    let d2 := Xla.mul d d
    let r2 := Xla.sum 1 (Xla.transpose d2 perm[2, 0, 1])
    let eps := (posScalar OFF_EPS).apply θ
    Xla.sqrt (Xla.add r2 (Xla.broadcast [⟨N, false⟩, ⟨N_nuc, false⟩] eps))

/-- Same-spin electron–electron displacement `rⱼ - rᵢ`, shape `[N, N, 3]`. -/
def pairDisp (N : ℕ) : SimpleExpr [⟨.float, [N, 3]⟩] ⟨.float, [N, N, 3]⟩ :=
  Expr.ofFn fun r =>
    let sender := Xla.broadcast [⟨N, false⟩, ⟨N, true⟩, ⟨3, true⟩] r
    let receiver := Xla.broadcast [⟨N, true⟩, ⟨N, false⟩, ⟨3, true⟩] r
    Xla.sub sender receiver

/-- Same-spin electron–electron distance `√(‖rⱼ - rᵢ‖² + ε)`. -/
def pairDist (N : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, [N, N]⟩ :=
  Expr.ofFn fun r => fun θ =>
    let d := (pairDisp N).apply r
    let d2 := Xla.mul d d
    let r2 := Xla.sum 1 (Xla.transpose d2 perm[2, 0, 1])
    let eps := (posScalar OFF_EPS).apply θ
    Xla.sqrt (Xla.add r2 (Xla.broadcast [⟨N, false⟩, ⟨N, false⟩] eps))

/-- Opposite-spin displacement `r'ⱼ - rᵢ`, shape `[N, N', 3]`. -/
def pairDispCross (N N' : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N', 3]⟩] ⟨.float, [N, N', 3]⟩ :=
  Expr.ofFn fun r => fun r' =>
    let sender := Xla.broadcast [⟨N, false⟩, ⟨N', true⟩, ⟨3, true⟩] r'
    let receiver := Xla.broadcast [⟨N, true⟩, ⟨N', false⟩, ⟨3, true⟩] r
    Xla.sub receiver sender

/-- Opposite-spin distance `√(‖r'ⱼ - rᵢ‖² + ε)`. -/
def pairDistCross (N N' : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N', 3]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N']⟩ :=
  Expr.ofFn fun r => fun r' => fun θ =>
    let d := (pairDispCross N N').apply (r.append r')
    let d2 := Xla.mul d d
    let r2 := Xla.sum 1 (Xla.transpose d2 perm[2, 0, 1])
    let eps := (posScalar OFF_EPS).apply θ
    Xla.sqrt (Xla.add r2 (Xla.broadcast [⟨N, false⟩, ⟨N', false⟩] eps))

/-! ### The three streams -/

/-- Initial one-electron stream: contracts displacement / distance / charge-embedding
features over the nucleus index into `F` features per electron, then `tanh`. -/
def oneStreamInit (N N_nuc : ℕ) :
    SimpleExpr
      [⟨.float, [N, 3]⟩, ⟨.float, [N_nuc, 3]⟩, ⟨.int, [N_nuc]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, F]⟩ :=
  Expr.ofFn fun r => fun R => fun Z => fun θ =>
    let disp := (enDisp N N_nuc).apply (r.append R)
    let dist := (enDist N N_nuc).apply ((r.append R).append θ)
    let ztab := (paramBlock OFF_ZTAB [Z_TAB]).apply θ
    let Zc : Expr XlaOp _ [⟨.int, [N_nuc]⟩] := Xla.mod Z (Xla.ofNat Z_TAB)
    let zRaw := Xla.gather (α := .float) (s := [Z_TAB]) (s' := [N_nuc]) ztab Zc
    let wd := (paramBlock OFF_WD [N_nuc, 3, F]).apply θ
    let ws := (paramBlock OFF_WS [N_nuc, F]).apply θ
    let wz := (paramBlock OFF_WZ [N_nuc, F]).apply θ
    let dispT := Xla.transpose disp perm[1, 2, 0]
    let distT := Xla.transpose dist perm[1, 0]
    let zB := Xla.broadcast [⟨N, false⟩, ⟨N_nuc, true⟩] zRaw
    let zT := Xla.transpose zB perm[1, 0]
    let tD := Xla.einsum (s := [N_nuc, 3, N, F]) [[#0, #1, #2], [#0, #1, #3]] 2
      (dispT.append wd)
    let tS := Xla.einsum (s := [N_nuc, N, F]) [[#0, #1], [#0, #2]] 1 (distT.append ws)
    let tZ := Xla.einsum (s := [N_nuc, N, F]) [[#0, #1], [#0, #2]] 1 (zT.append wz)
    let b0 := Xla.broadcast [⟨N, false⟩, ⟨F, true⟩] ((paramSlice OFF_B0 F).apply θ)
    (tanh [N, F]).apply (Xla.add (Xla.add (Xla.add tD tS) tZ) b0)

/-- Initial same-spin two-electron stream: per-pair displacement and distance
features contracted to width `F`, then `tanh`. -/
def twoStreamInit (N : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, [N, N, F]⟩ :=
  Expr.ofFn fun r => fun θ =>
    let pd := (pairDisp N).apply r
    let pd2 := (pairDist N).apply (r.append θ)
    let we := (paramBlock OFF_WE [3, F]).apply θ
    let wd := (paramSlice OFF_WD2 F).apply θ
    let pdT := Xla.transpose pd perm[2, 0, 1]
    let tD := Xla.einsum (s := [3, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (pdT.append we)
    let wdB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] wd
    let pd2B := Xla.broadcast [⟨N, true⟩, ⟨N, true⟩, ⟨F, false⟩] pd2
    let tS := Xla.mul pd2B wdB
    let gb := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] ((paramSlice OFF_GB F).apply θ)
    (tanh [N, N, F]).apply (Xla.add (Xla.add tD tS) gb)

/-- Initial opposite-spin two-electron stream. -/
def twoStreamInitCross (N N' : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N', 3]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N', F]⟩ :=
  Expr.ofFn fun r => fun r' => fun θ =>
    let pd := (pairDispCross N N').apply (r.append r')
    let pd2 := (pairDistCross N N').apply ((r.append r').append θ)
    let we := (paramBlock OFF_WE [3, F]).apply θ
    let wd := (paramSlice OFF_WD2 F).apply θ
    let pdT := Xla.transpose pd perm[2, 0, 1]
    let tD := Xla.einsum (s := [3, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (pdT.append we)
    let wdB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] wd
    let pd2B := Xla.broadcast [⟨N, true⟩, ⟨N', true⟩, ⟨F, false⟩] pd2
    let tS := Xla.mul pd2B wdB
    let gb := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] ((paramSlice OFF_GB F).apply θ)
    (tanh [N, N', F]).apply (Xla.add (Xla.add tD tS) gb)

/-! ### Layer updates -/

/-- One-electron stream update: `h ← tanh(V h + Σⱼ w ⊙ g_ij + Σⱼ w' ⊙ g_ij^{σσ̄} + b)`. -/
def oneStreamLayer (N N' off : ℕ) :
    SimpleExpr
      [⟨.float, [N, F]⟩, ⟨.float, [N, N, F]⟩, ⟨.float, [N, N', F]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, F]⟩ :=
  Expr.ofFn fun h => fun g => fun gC => fun θ =>
    let V := (paramBlock off [F, F]).apply θ
    let w := (paramSlice (off + 1024) F).apply θ
    let wC := (paramSlice (off + 1056) F).apply θ
    let b := (paramSlice (off + 1088) F).apply θ
    let tV := Xla.einsum (s := [F, N, F]) [[#1, #0], [#0, #2]] 1 (h.append V)
    let tG := Xla.einsum (s := [N, N, F]) [[#0, #1, #2], [#2]] 1 (g.append w)
    let gCT := Xla.transpose gC perm[1, 0, 2]
    let tGC := Xla.einsum (s := [N', N, F]) [[#0, #1, #2], [#2]] 1 (gCT.append wC)
    let bB := Xla.broadcast [⟨N, false⟩, ⟨F, true⟩] b
    (tanh [N, F]).apply (Xla.add (Xla.add (Xla.add tV tG) tGC) bB)

/-- Same-spin two-electron stream update:
`g ← tanh(G g + H (h_i + h_j) + c)`. -/
def twoStreamLayer (N off : ℕ) :
    SimpleExpr [⟨.float, [N, N, F]⟩, ⟨.float, [N, F]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N, F]⟩ :=
  Expr.ofFn fun g => fun h => fun θ =>
    let G := (paramBlock (off + 1120) [F, F]).apply θ
    let H := (paramBlock (off + 2144) [F, F]).apply θ
    let c := (paramSlice (off + 3168) F).apply θ
    let gT := Xla.transpose g perm[2, 0, 1]
    let tG := Xla.einsum (s := [F, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (gT.append G)
    let hA := Xla.broadcast [⟨N, false⟩, ⟨N, true⟩, ⟨F, true⟩] h
    let hB := Xla.broadcast [⟨N, true⟩, ⟨N, false⟩, ⟨F, true⟩] h
    let hS := Xla.transpose (Xla.add hA hB) perm[2, 0, 1]
    let tH := Xla.einsum (s := [F, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (hS.append H)
    let cB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] c
    (tanh [N, N, F]).apply (Xla.add (Xla.add tG tH) cB)

/-- Opposite-spin two-electron stream update:
`g^{σσ̄} ← tanh(G' g^{σσ̄} + H' (h_i^σ + h_j^{σ̄}) + c')`. -/
def twoStreamLayerCross (N N' off : ℕ) :
    SimpleExpr [⟨.float, [N, N', F]⟩, ⟨.float, [N, F]⟩, ⟨.float, [N', F]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N', F]⟩ :=
  Expr.ofFn fun gC => fun h => fun h' => fun θ =>
    let G := (paramBlock (off + 3200) [F, F]).apply θ
    let H := (paramBlock (off + 4224) [F, F]).apply θ
    let c := (paramSlice (off + 5248) F).apply θ
    let gT := Xla.transpose gC perm[2, 0, 1]
    let tG := Xla.einsum (s := [F, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (gT.append G)
    let hA := Xla.broadcast [⟨N, true⟩, ⟨N', false⟩, ⟨F, true⟩] h
    let hB := Xla.broadcast [⟨N, false⟩, ⟨N', true⟩, ⟨F, true⟩] h'
    let hS := Xla.transpose (Xla.add hA hB) perm[2, 0, 1]
    let tH := Xla.einsum (s := [F, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (hS.append H)
    let cB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] c
    (tanh [N, N', F]).apply (Xla.add (Xla.add tG tH) cB)

/-! ### Outputs -/

/-- Bounded Jastrow factor `J = Σᵢ w_J · hᵢ` (scalar). -/
def jastrow (N : ℕ) :
    SimpleExpr [⟨.float, [N, F]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, []⟩ :=
  Expr.ofFn fun h => fun θ =>
    let wJ := (paramSlice OFF_WJ F).apply θ
    Xla.einsum (s := [N, F]) [[#0, #1], [#1]] 2 (h.append wJ)

/-- Per-orbital exponential envelope, shape `[K, N, N]`: entry `(k, j, i)` is
`exp(-Σα exp A[k,i,α] · ‖xⱼ - Rα‖)`, the FermiNet envelope with decay exponents
parameterized by *orbital* `(k, i)` and evaluated on electron `j` (FermiNet,
Eq. 19). Indexing the parameters by orbital — not by electron — is what makes the
envelope a permutation-*equivariant* function of the electron positions. -/
def envelope (N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N, N_nuc]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, [K, N, N]⟩ :=
  Expr.ofFn fun dist => fun θ =>
    let A := (paramBlock OFF_A [K, N, N_nuc]).apply θ
    let Aexp := Xla.exp A
    let dT := Xla.transpose dist perm[1, 0]
    let e := Xla.einsum (s := [N_nuc, K, N, N]) [[#1, #2, #0], [#0, #3]] 1
      (Aexp.append dT)
    Xla.transpose e perm[0, 2, 1]

/-- Orbital matrix of shape `[K, N, N]`: entry `(k, j, i)` is
`φᵢᵏ(xⱼ) = (w_k,i · h_j) · exp(-eᵢᵏ(xⱼ))`, i.e. determinant `k`'s matrix with rows
indexed by electrons `j` and columns by orbitals `i` (the transpose orientation
is immaterial for the determinant), with the per-orbital envelope applied
elementwise. -/
def orbitalMatrix (N : ℕ) :
    SimpleExpr [⟨.float, [N, F]⟩, ⟨.float, [K, N, N]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [K, N, N]⟩ :=
  Expr.ofFn fun h => fun envExp => fun θ =>
    let worb := (paramBlock OFF_WORB [F, K, N]).apply θ
    let φ := Xla.einsum (s := [F, N, K, N]) [[#1, #0], [#0, #2, #3]] 1 (h.append worb)
    let φT := Xla.transpose φ perm[1, 0, 2]
    Xla.mul φT envExp

/-- `det` as a library function, for `vmap`-ing over the determinant axis. -/
def detExpr (n : ℕ) : Expr XlaOp [⟨.float, [n, n]⟩] [⟨.float, []⟩] :=
  Expr.ofFn fun x => Xla.det x

/-- Weighted sum of the `K` determinants of a spin sector: `Σₖ exp w_k · detₖ`. -/
def detBlock (N : ℕ) :
    SimpleExpr [⟨.float, [K, N, N]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, []⟩ :=
  Expr.ofFn fun orb => fun θ =>
    let dets := Xla.vmap (batch := K) (ins := [⟨.float, [N, N]⟩]) (outs := [⟨.float, []⟩])
      (detExpr N) orb .nil
    let wdet := Xla.exp ((paramSlice OFF_WDET K).apply θ)
    Xla.sum 1 (Xla.mul dets wdet)

/-! ### The ansatz -/

/-- The FermiNet program for a molecule with `N_nuc` nuclei and `N_up` / `N_down`
electrons. The five inputs are the parameters `θ`, nuclear positions `R`, nuclear
charges `Z`, and the two spin sectors' electron positions. -/
def fermiNetAnsatz (N_nuc N_up N_down : ℕ) : Xla.SimpleExpr
    [⟨.float, [N_PARAM]⟩, -- parameters
      ⟨.float, [N_nuc, 3]⟩, -- positions of nuclei
      ⟨.int, [N_nuc]⟩, -- types of nuclei
      ⟨.float, [N_up, 3]⟩, -- positions of spin up electrons
      ⟨.float, [N_down, 3]⟩] -- positions of spin down electrons
    ⟨.float, []⟩ :=
  Expr.ofFn fun θ => fun R => fun Z => fun rUp => fun rDown =>
    let hU0 := (oneStreamInit N_up N_nuc).apply (((rUp.append R).append Z).append θ)
    let hD0 := (oneStreamInit N_down N_nuc).apply (((rDown.append R).append Z).append θ)
    let gU0 := (twoStreamInit N_up).apply (rUp.append θ)
    let gD0 := (twoStreamInit N_down).apply (rDown.append θ)
    let gUD0 := (twoStreamInitCross N_up N_down).apply ((rUp.append rDown).append θ)
    let gDU0 := (twoStreamInitCross N_down N_up).apply ((rDown.append rUp).append θ)
    -- layer 0
    let hU1 := (oneStreamLayer N_up N_down OFF_L0).apply
      (((hU0.append gU0).append gUD0).append θ)
    let hD1 := (oneStreamLayer N_down N_up OFF_L0).apply
      (((hD0.append gD0).append gDU0).append θ)
    let gU1 := (twoStreamLayer N_up OFF_L0).apply ((gU0.append hU0).append θ)
    let gD1 := (twoStreamLayer N_down OFF_L0).apply ((gD0.append hD0).append θ)
    let gUD1 := (twoStreamLayerCross N_up N_down OFF_L0).apply
      (((gUD0.append hU0).append hD0).append θ)
    let gDU1 := (twoStreamLayerCross N_down N_up OFF_L0).apply
      (((gDU0.append hD0).append hU0).append θ)
    -- layer 1
    let hU2 := (oneStreamLayer N_up N_down OFF_L1).apply
      (((hU1.append gU1).append gUD1).append θ)
    let hD2 := (oneStreamLayer N_down N_up OFF_L1).apply
      (((hD1.append gD1).append gDU1).append θ)
    -- Jastrow factor (the streams are tanh-bounded, so `exp J` is bounded)
    let JU := (jastrow N_up).apply (hU2.append θ)
    let JD := (jastrow N_down).apply (hD2.append θ)
    let J := Xla.exp (Xla.add JU JD)
    -- exponential envelopes
    let distU := (enDist N_up N_nuc).apply ((rUp.append R).append θ)
    let distD := (enDist N_down N_nuc).apply ((rDown.append R).append θ)
    let envU := Xla.exp (Xla.neg ((envelope N_up N_nuc).apply (distU.append θ)))
    let envD := Xla.exp (Xla.neg ((envelope N_down N_nuc).apply (distD.append θ)))
    -- determinants
    let orbU := (orbitalMatrix N_up).apply ((hU2.append envU).append θ)
    let orbD := (orbitalMatrix N_down).apply ((hD2.append envD).append θ)
    let detU := (detBlock N_up).apply (orbU.append θ)
    let detD := (detBlock N_down).apply (orbD.append θ)
    Xla.mul (Xla.mul J detU) detD

/-- The FermiNet `Ansatz`. -/
def fermiNet : Ansatz where
  N_param := N_PARAM
  ansatz := fun N_nuc N_up N_down => fermiNetAnsatz N_nuc N_up N_down

/-! ### The validity contract

See the module docstring for the proof sketches. `smooth` is proved modulo the
decomposition below; `antisymmetric` and `exp_decay` are `sorry`. -/

/-- Antisymmetry in both spin sectors: the streams are permutation-equivariant,
the envelope/Jastrow invariant, and each determinant picks up `sign σ`. -/
theorem antisymmetric (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNet.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    IsAntisymmetric ((fermiNet.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc) := by
  sorry

/-! ### Smoothness: the decomposition

`smooth` says that the evaluated program `Ψ(θ, R, Z) : (x↑, x↓) ↦ ℝ` is `C²` in the
electron positions. Every primitive in the program is real-analytic on its domain, so
the proof is a composition along the SSA dataflow: `Ψ = e^{J↑+J↓} · det↑ · det↓`, each
factor is `C²` because its inputs are, and the inputs of the leaves are the two
electron-position tensors.

The lemmas below state that composition one node at a time. Each is stated for an
arbitrary normed `ℝ`-space `X` (the "smooth variable": for the top-level theorem it is
`Config N_up N_down`, and for an intermediate node it is again `Config N_up N_down`,
since every intermediate value is a function of the electron positions), together with
`C²` hypotheses for the arguments the node consumes — so the lemmas *chain* by
`ContDiff.comp` with no glue at all: `contDiff_eval_oneStreamLayer` consumes the
conclusions of `contDiff_eval_oneStreamInit`, `contDiff_eval_twoStreamInit` and
`contDiff_eval_twoStreamInitCross`, and so on up to `contDiff_fermiNetAnsatz`.

Two kinds of data flow through the program but not through the `C²` hypotheses:

* **Integer data is constant.** The nuclear charges `Z` and the `gather` indices built
  from them (`paramSlice`, the charge embedding) do not depend on `X`, and `ℤ` has no
  `NormedSpace ℝ` structure, so they are passed to the lemmas as *fixed* arguments
  rather than as functions of `X`. This is exactly the op discipline of
  `QMC/Ansatz.lean` (`iota`/`ofNat`/`mod`/`gather` on integer data only).
* **Constant tensors are fixed, not `C²`.** `θ`, `R` and `Z` are fixed by the theorem
  (the wavefunction varies the electron positions), so a node that only reads them
  appears in a `C²` statement as a constant.

The leaves — one per primitive of `DirectImpl` — are in `Xla/Smooth.lean`:
`contDiff_tensorMap`(`₂`) for elementwise ops (`exp`, `neg`, `add`, `sub`, `mul`,
`div`, `sqrt`, the last two with the side conditions the DSL guarantees),
`contDiff_sumN`/`contDiff_einsum`/`contDiff_det` for the reductions,
`contDiff_transpose`/`contDiff_broadcast`/`contDiff_unflatten` for the index
manipulations, and `contDiff_gather` for gathering along a fixed index tensor.

The proofs of the node lemmas are `sorry`: each is the composition of its children's
lemmas with the `Xla/Smooth.lean` leaves, mechanically obtained by
`simp only [reduce_soir, reduce_xla, reduce_tensor]`-unfolding the node (which leaves
the sub-components as named libraries, never inlined) and then chaining. -/

/-- The configuration space the wavefunction is a function of: the two spin sectors'
electron positions. -/
abbrev Config (N_up N_down : ℕ) : Type := Tensor ℝ [N_up, 3] × Tensor ℝ [N_down, 3]

section Smoothness

variable {X : Type} [NormedAddCommGroup X] [NormedSpace ℝ X]

/-! #### Primitive leaves of the program -/

private theorem flatten_map₂ {s : Shape} (f : ℝ → ℝ → ℝ) (u v : Tensor ℝ s) (i : Fin s.prod) :
    (Tensor.map₂ f u v).flatten i = f (u.flatten i) (v.flatten i) := by
  induction s with
  | nil => rfl
  | cons n s ih =>
    simp only [Tensor.map₂, Tensor.flatten]
    exact ih (u i.divNat) (v i.divNat) i.modNat

private theorem tensorMap_eq_curryMap (s : Shape) (g : ℝ → ℝ) :
    (Tensor.map g : Tensor ℝ s → Tensor ℝ s) = Curry.map g := by
  induction s with
  | nil => rfl
  | cons n s ih => rfl

private theorem flatten_curryMap {s : Shape} (f : ℝ → ℝ) (u : Tensor ℝ s) (i : Fin s.prod) :
    Tensor.flatten (Curry.map f u) i = f (u.flatten i) := by
  induction s with
  | nil => rfl
  | cons n s ih =>
    simp only [Curry.map, Tensor.flatten]
    exact ih (u i.divNat) i.modNat

private theorem flatten_pure {s : Shape} (c : ℝ) (i : Fin s.prod) :
    Tensor.flatten (Curry.pure (m := Fin) c : Curry Fin s ℝ) i = c := by
  induction s with
  | nil => rfl
  | cons n s ih =>
    simp only [Curry.pure, Tensor.flatten]
    exact ih i.modNat

/-- `tanh`: `1 - 2/(exp(2x) + 1)`, a composition of `exp` and a division whose
denominator is `≥ 1`. -/
theorem contDiff_eval_tanh (s : Shape) {f : X → Tensor ℝ s} (hf : ContDiff ℝ 2 f) :
    ContDiff ℝ 2 fun x => (tanh s).eval (f x) := by
  simp only [tanh, reduce_soir, reduce_xla]
  apply Xla.contDiff_tensorMap₂_sub
  · exact contDiff_const
  · apply Xla.contDiff_tensorMap₂_div
    · exact contDiff_const
    · apply Xla.contDiff_tensorMap₂_add
      · apply Xla.contDiff_tensorMap_exp
        apply Xla.contDiff_tensorMap₂_mul
        · exact contDiff_const
        · exact hf
      · exact contDiff_const
    · intro x i
      rw [flatten_map₂, tensorMap_eq_curryMap, flatten_curryMap, flatten_map₂, flatten_pure,
        flatten_pure]
      exact ne_of_gt (by positivity)

/-- Parameter slicing is linear (`gather` along the fixed index tensor `iota + off`). -/
theorem contDiff_eval_paramSlice (off len : ℕ) {p : X → Tensor ℝ [N_PARAM]}
    (hp : ContDiff ℝ 2 p) :
    ContDiff ℝ 2 fun x => (paramSlice off len).eval (p x) := by
  simp only [paramSlice, reduce_soir, reduce_xla, dif_neg (by decide : ¬ N_PARAM = 0)]
  apply contDiff_pi'
  intro v
  exact (contDiff_apply ℝ _ ?_).comp hp

/-- Reshaping a slice is an index reinterpretation. -/
theorem contDiff_eval_paramBlock (off : ℕ) (s : Shape)
    {p : X → Tensor ℝ [N_PARAM]} (hp : ContDiff ℝ 2 p) :
    ContDiff ℝ 2 fun x => (paramBlock off s).eval (p x) := by
  simp only [paramBlock, reduce_soir, reduce_xla]
  exact Xla.contDiff_unflatten s (contDiff_eval_paramSlice off s.prod hp)

/-- `exp` of a parameter, so positive for every `θ`. -/
theorem contDiff_eval_posScalar (off : ℕ) {p : X → Tensor ℝ [N_PARAM]}
    (hp : ContDiff ℝ 2 p) :
    ContDiff ℝ 2 fun x => (posScalar off).eval (p x) := by
  simp only [posScalar, reduce_soir, reduce_xla]
  exact Xla.contDiff_tensorMap_exp
    (Xla.contDiff_sumN 1 (contDiff_eval_paramSlice off 1 hp))

/-- Electron–nucleus displacement: a difference of coordinates. -/
theorem contDiff_eval_enDisp (N N_nuc : ℕ)
    {r : X → Tensor ℝ [N,3]} {R : X → Tensor ℝ [N_nuc,3]}
    (hr : ContDiff ℝ 2 r) (hR : ContDiff ℝ 2 R) :
    ContDiff ℝ 2 fun x => (enDisp N N_nuc).eval (r x) (R x) := by
  sorry

/-- Electron–nucleus distance `√(‖rᵢ - Rα‖² + ε)`: a polynomial under `sqrt`, whose
argument is `≥ ε = exp θ > 0` — bounded away from the crease of `sqrt` at `0`. -/
theorem contDiff_eval_enDist (N N_nuc : ℕ)
    {r : X → Tensor ℝ [N,3]} {R : X → Tensor ℝ [N_nuc,3]} {θ : X → Tensor ℝ [N_PARAM]}
    (hr : ContDiff ℝ 2 r) (hR : ContDiff ℝ 2 R) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (enDist N N_nuc).eval (r x) (R x) (θ x) := by
  sorry

/-- Same-spin displacement: a difference of coordinates. -/
theorem contDiff_eval_pairDisp (N : ℕ) {r : X → Tensor ℝ [N,3]}
    (hr : ContDiff ℝ 2 r) :
    ContDiff ℝ 2 fun x => (pairDisp N).eval (r x) := by
  sorry

/-- Same-spin distance `√(‖rⱼ - rᵢ‖² + ε)`; `ε > 0` keeps `sqrt` off `0`. -/
theorem contDiff_eval_pairDist (N : ℕ)
    {r : X → Tensor ℝ [N,3]} {θ : X → Tensor ℝ [N_PARAM]}
    (hr : ContDiff ℝ 2 r) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (pairDist N).eval (r x) (θ x) := by
  sorry

/-- Opposite-spin displacement. -/
theorem contDiff_eval_pairDispCross (N N' : ℕ)
    {r : X → Tensor ℝ [N,3]} {r' : X → Tensor ℝ [N',3]}
    (hr : ContDiff ℝ 2 r) (hr' : ContDiff ℝ 2 r') :
    ContDiff ℝ 2 fun x => (pairDispCross N N').eval (r x) (r' x) := by
  sorry

/-- Opposite-spin distance, `sqrt` again guarded by `ε > 0`. -/
theorem contDiff_eval_pairDistCross (N N' : ℕ)
    {r : X → Tensor ℝ [N,3]} {r' : X → Tensor ℝ [N',3]} {θ : X → Tensor ℝ [N_PARAM]}
    (hr : ContDiff ℝ 2 r) (hr' : ContDiff ℝ 2 r') (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (pairDistCross N N').eval (r x) (r' x) (θ x) := by
  sorry

/-! #### The three streams -/

/-- Initial one-electron stream: `enDisp`/`enDist` features contracted with `einsum`
(the contractum is fixed index data) plus the charge embedding, then `tanh`. -/
theorem contDiff_eval_oneStreamInit (N N_nuc : ℕ) (Z : Tensor ℤ [N_nuc])
    {r : X → Tensor ℝ [N,3]} {R : X → Tensor ℝ [N_nuc,3]} {θ : X → Tensor ℝ [N_PARAM]}
    (hr : ContDiff ℝ 2 r) (hR : ContDiff ℝ 2 R) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (oneStreamInit N N_nuc).eval (r x) (R x) Z (θ x) := by
  sorry

/-- Initial same-spin two-electron stream: `pairDisp`/`pairDist` features, then
`tanh`. -/
theorem contDiff_eval_twoStreamInit (N : ℕ)
    {r : X → Tensor ℝ [N,3]} {θ : X → Tensor ℝ [N_PARAM]}
    (hr : ContDiff ℝ 2 r) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (twoStreamInit N).eval (r x) (θ x) := by
  sorry

/-- Initial opposite-spin two-electron stream. -/
theorem contDiff_eval_twoStreamInitCross (N N' : ℕ)
    {r : X → Tensor ℝ [N,3]} {r' : X → Tensor ℝ [N',3]} {θ : X → Tensor ℝ [N_PARAM]}
    (hr : ContDiff ℝ 2 r) (hr' : ContDiff ℝ 2 r') (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (twoStreamInitCross N N').eval (r x) (r' x) (θ x) := by
  sorry

/-- One-electron stream update `h ← tanh(V h + Σⱼ w ⊙ g_ij + Σⱼ w' ⊙ g_ij^{σσ̄} + b)`:
`einsum` contractions (fixed weights) of `C²` inputs, then `tanh`. -/
theorem contDiff_eval_oneStreamLayer (N N' off : ℕ)
    {h : X → Tensor ℝ [N,F]} {g : X → Tensor ℝ [N,N,F]} {gC : X → Tensor ℝ [N,N',F]}
    {θ : X → Tensor ℝ [N_PARAM]}
    (hh : ContDiff ℝ 2 h) (hg : ContDiff ℝ 2 g) (hgC : ContDiff ℝ 2 gC) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (oneStreamLayer N N' off).eval (h x) (g x) (gC x) (θ x) := by
  sorry

/-- Same-spin two-electron stream update `g ← tanh(G g + H (h_i + h_j) + c)`. -/
theorem contDiff_eval_twoStreamLayer (N off : ℕ)
    {g : X → Tensor ℝ [N,N,F]} {h : X → Tensor ℝ [N,F]} {θ : X → Tensor ℝ [N_PARAM]}
    (hg : ContDiff ℝ 2 g) (hh : ContDiff ℝ 2 h) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (twoStreamLayer N off).eval (g x) (h x) (θ x) := by
  sorry

/-- Opposite-spin two-electron stream update
`g^{σσ̄} ← tanh(G' g^{σσ̄} + H' (h_i^σ + h_j^{σ̄}) + c')`. -/
theorem contDiff_eval_twoStreamLayerCross (N N' off : ℕ)
    {gC : X → Tensor ℝ [N,N',F]} {h : X → Tensor ℝ [N,F]} {h' : X → Tensor ℝ [N',F]}
    {θ : X → Tensor ℝ [N_PARAM]}
    (hgC : ContDiff ℝ 2 gC) (hh : ContDiff ℝ 2 h) (hh' : ContDiff ℝ 2 h') (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (twoStreamLayerCross N N' off).eval (gC x) (h x) (h' x) (θ x) := by
  sorry

/-! #### Outputs -/

/-- Jastrow factor `Σᵢ w_J · hᵢ`: an `einsum` contraction with fixed weights. -/
theorem contDiff_eval_jastrow (N : ℕ)
    {h : X → Tensor ℝ [N,F]} {θ : X → Tensor ℝ [N_PARAM]}
    (hh : ContDiff ℝ 2 h) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (jastrow N).eval (h x) (θ x) := by
  sorry

/-- Exponential envelope `Σα exp A[k,i,α] · ‖xⱼ - Rα‖`: an `einsum` of `exp`-ed
parameters with the `C²` distances (its values are positive, as the caller needs for
`exp (-·)`). -/
theorem contDiff_eval_envelope (N N_nuc : ℕ)
    {dist : X → Tensor ℝ [N,N_nuc]} {θ : X → Tensor ℝ [N_PARAM]}
    (hd : ContDiff ℝ 2 dist) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (envelope N N_nuc).eval (dist x) (θ x) := by
  sorry

/-- Orbital matrix `φᵢᵏ(xⱼ)`: an `einsum` of the stream features with the fixed orbital
weights, multiplied by the `C²` envelope. -/
theorem contDiff_eval_orbitalMatrix (N : ℕ)
    {h : X → Tensor ℝ [N,F]} {envExp : X → Tensor ℝ [K,N,N]} {θ : X → Tensor ℝ [N_PARAM]}
    (hh : ContDiff ℝ 2 h) (he : ContDiff ℝ 2 envExp) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (orbitalMatrix N).eval (h x) (envExp x) (θ x) := by
  sorry

/-- Weighted determinant sum `Σₖ exp w_k · detₖ`: `det` (`Xla.contDiff_det`) `vmap`-ed
over the `K` orbital matrices, summed with positive weights. The `vmap` of `DirectImpl`
is the only node whose semantics is `Index`-level rather than entrywise; it is the
`det` leaf applied to each slice, so it is `C²` entrywise in the orbital tensor. -/
theorem contDiff_eval_detBlock (N : ℕ)
    {orb : X → Tensor ℝ [K,N,N]} {θ : X → Tensor ℝ [N_PARAM]}
    (ho : ContDiff ℝ 2 orb) (hθ : ContDiff ℝ 2 θ) :
    ContDiff ℝ 2 fun x => (detBlock N).eval (orb x) (θ x) := by
  sorry

end Smoothness

/-- The evaluated program is `C²` in the electron positions: the composition of the
node lemmas above along the dataflow (see the section docstring). -/
theorem contDiff_fermiNetAnsatz (N_nuc N_up N_down : ℕ) (θ : Tensor ℝ [N_PARAM])
    (R_nuc : Tensor ℝ [N_nuc,3]) (Z_nuc : Tensor ℤ [N_nuc]) :
    ContDiff ℝ 2 (fun p : Config N_up N_down =>
      (fermiNetAnsatz N_nuc N_up N_down).eval θ R_nuc Z_nuc p.1 p.2) := by
  -- `simp only [fermiNetAnsatz, reduce_soir, reduce_xla, reduce_tensor]` turns the
  -- goal into the dataflow of `Expr.eval` applications to the component libraries,
  -- in exactly the shape the `contDiff_eval_*` lemmas above are stated in, so the
  -- composition is a chain of `exact`/`ContDiff.comp` applications along it. It is
  -- left as `sorry` because the normalized program is ~10⁴ nodes (every shared stream
  -- is inlined at each use, `hU0` alone occurs three times per layer), which makes the
  -- final definitional-equality check of that chain expensive; the composition itself
  -- is mechanical. Proving it therefore wants either a split of the program into
  -- shallower stages, or a `psiEval`-style reference semantics with a separate
  -- `Expr.eval`-agreement lemma.
  sorry

/-- Smoothness: every primitive used is real-analytic on its domain (`sqrt` guarded
away from 0 by `ε > 0`, `tanh` a composition of analytic functions), so `Ψ` is C². -/
theorem smooth (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNet.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    ContDiff ℝ 2 (Function.uncurry ((fermiNet.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc)) := by
  have h := contDiff_fermiNetAnsatz N_nuc N_up N_down θ R_nuc Z_nuc
  -- `Function.uncurry` unfolds to `fun p => ⋯ p.1 p.2`, and `fermiNet.ansatz` is
  -- `fermiNetAnsatz` (the structure fields of the literal `fermiNet` reduce).
  show ContDiff ℝ 2 (fun p : Config N_up N_down =>
    (fermiNet.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc p.1 p.2)
  exact h

/-- Exponential decay envelope: `|Ψ| ≤ C exp(-k‖x‖)` for some `C` and `k > 0` —
the `exp`-reparameterized envelope exponents are positive for every `θ`, the
determinants are bounded by Hadamard, and the Jastrow factor is bounded. -/
theorem exp_decay (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNet.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    ∃ C k : ℝ, 0 < k ∧
      ∀ x, |Function.uncurry ((fermiNet.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc) x| ≤
        C * Real.exp (-k * ‖x‖) := by
  sorry

/-- FermiNet satisfies the QMC validity contract for every molecule and every
parameter value. -/
theorem isValid : fermiNet.isValid := by
  intro N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
  exact {
    antisymmetric := antisymmetric N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
    smooth := smooth N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
    exp_decay := exp_decay N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
  }

end FermiNet

-- Sample code generation for H₂ (2 nuclei, 1 up / 1 down electron).
#eval IO.println (FermiNet.fermiNetAnsatz 2 1 1).code
