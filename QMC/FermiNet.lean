import QMC.Ansatz

/-!
# FermiNet — a neural-network QMC ansatz for molecules

FermiNet (Pfau, Spencer, Matthews, Foulkes, *Ab initio solution of the many-electron
Schrödinger equation with deep neural networks*, Phys. Rev. Research 2, 033429
(2020)) is the canonical neural network wavefunction for molecules and the
architectural ancestor of the current state-of-the-art QMC ansätze (Moon,
PsiFormer, PauliNet, FermiNet with Pfaffians). This file implements the FermiNet
architecture as a single `Xla.SimpleExpr` program in the Soir/XLA DSL, packaged as
an `Ansatz` satisfying the contract of `QMC/Ansatz.lean`.

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
  *every* `θ` (the `∀θ` contract).
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
* parameter slicing for smaller molecules is `gather` over
  `iota + const` indices; a dedicated slice primitive would be cleaner.
-/

namespace FermiNet

open Soir
open Xla

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

section

variable {args : List TensorType}

/-! ### Primitives composed from the existing op set -/

/-- `tanh` from primitives: `1 - 2 / (exp (2x) + 1)`. Mathematically exact and
numerically stable for both signs of overflow. -/
def tanh {s : Shape} (x : Expr XlaOp args [⟨.float, s⟩]) :
    Expr XlaOp args [⟨.float, s⟩] :=
  let two : Expr XlaOp args [⟨.float, s⟩] := Xla.ofNat 2
  let one : Expr XlaOp args [⟨.float, s⟩] := Xla.ofNat 1
  let denom := Xla.add (Xla.exp (Xla.mul two x)) one
  Xla.sub one (Xla.div two denom)

/-! ### Parameter access -/

/-- Slice `len` consecutive floats from the parameter vector starting at `off`. -/
def paramSlice (off len : ℕ) (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [len]⟩] :=
  let idx : Expr XlaOp args [⟨.int, [len]⟩] :=
    Xla.add (Xla.iota len) (Xla.ofNat off)
  Xla.gather (α := .float) (s := [N_PARAM]) (s' := [len]) θ idx

/-- Reshape a slice of the parameter vector into a tensor of shape `s`. -/
def paramBlock (off : ℕ) (s : Shape) (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, s⟩] :=
  Xla.unflatten s (paramSlice off s.prod θ)

/-- A scalar parameter reparameterized through `exp`, so it is `> 0` for every `θ`. -/
def posScalar (off : ℕ) (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, []⟩] :=
  Xla.exp (Xla.sum 1 (paramSlice off 1 θ))

/-! ### Geometry features -/

/-- Electron–nucleus displacement `rᵢ - Rα`, shape `[N, N_nuc, 3]`. -/
def enDisp (N N_nuc : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (R : Expr XlaOp args [⟨.float, [N_nuc, 3]⟩]) :
    Expr XlaOp args [⟨.float, [N, N_nuc, 3]⟩] :=
  let rB := Xla.broadcast [⟨N, true⟩, ⟨N_nuc, false⟩, ⟨3, true⟩] r
  let RB := Xla.broadcast [⟨N, false⟩, ⟨N_nuc, true⟩, ⟨3, true⟩] R
  Xla.sub rB RB

/-- Electron–nucleus distance `√(‖rᵢ - Rα‖² + ε)` with `ε = exp θ > 0`, keeping the
square root argument away from the non-analytic point 0. -/
def enDist (N N_nuc : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (R : Expr XlaOp args [⟨.float, [N_nuc, 3]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N_nuc]⟩] :=
  let d := enDisp N N_nuc r R
  let d2 := Xla.mul d d
  let d2T := Xla.transpose d2 perm[2, 0, 1]
  let r2 := Xla.sum 1 d2T
  let eps := posScalar OFF_EPS θ
  let epsB := Xla.broadcast [⟨N, false⟩, ⟨N_nuc, false⟩] eps
  Xla.sqrt (Xla.add r2 epsB)

/-- Same-spin electron–electron displacement `rⱼ - rᵢ`, shape `[N, N, 3]`. -/
def pairDisp (N : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩]) :
    Expr XlaOp args [⟨.float, [N, N, 3]⟩] :=
  let sender := Xla.broadcast [⟨N, false⟩, ⟨N, true⟩, ⟨3, true⟩] r
  let receiver := Xla.broadcast [⟨N, true⟩, ⟨N, false⟩, ⟨3, true⟩] r
  Xla.sub sender receiver

/-- Same-spin electron–electron distance `√(‖rⱼ - rᵢ‖² + ε)`. -/
def pairDist (N : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N]⟩] :=
  let d := pairDisp N r
  let d2 := Xla.mul d d
  let d2T := Xla.transpose d2 perm[2, 0, 1]
  let r2 := Xla.sum 1 d2T
  let eps := posScalar OFF_EPS θ
  let epsB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩] eps
  Xla.sqrt (Xla.add r2 epsB)

/-- Opposite-spin displacement `r'ⱼ - rᵢ`, shape `[N, N', 3]`. -/
def pairDispCross (N N' : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (r' : Expr XlaOp args [⟨.float, [N', 3]⟩]) :
    Expr XlaOp args [⟨.float, [N, N', 3]⟩] :=
  let sender := Xla.broadcast [⟨N, false⟩, ⟨N', true⟩, ⟨3, true⟩] r'
  let receiver := Xla.broadcast [⟨N, true⟩, ⟨N', false⟩, ⟨3, true⟩] r
  Xla.sub receiver sender

/-- Opposite-spin distance `√(‖r'ⱼ - rᵢ‖² + ε)`. -/
def pairDistCross (N N' : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (r' : Expr XlaOp args [⟨.float, [N', 3]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N']⟩] :=
  let d := pairDispCross N N' r r'
  let d2 := Xla.mul d d
  let d2T := Xla.transpose d2 perm[2, 0, 1]
  let r2 := Xla.sum 1 d2T
  let eps := posScalar OFF_EPS θ
  let epsB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩] eps
  Xla.sqrt (Xla.add r2 epsB)

/-! ### The three streams -/

/-- Initial one-electron stream: contracts displacement / distance / charge-embedding
features over the nucleus index into `F` features per electron, then `tanh`. -/
def oneStreamInit (N N_nuc : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (R : Expr XlaOp args [⟨.float, [N_nuc, 3]⟩])
    (Z : Expr XlaOp args [⟨.int, [N_nuc]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, F]⟩] :=
  let disp := enDisp N N_nuc r R
  let dist := enDist N N_nuc r R θ
  let ztab := paramBlock OFF_ZTAB [Z_TAB] θ
  let Zc : Expr XlaOp args [⟨.int, [N_nuc]⟩] :=
    Xla.mod Z (Xla.ofNat Z_TAB)
  let zRaw := Xla.gather (α := .float) (s := [Z_TAB]) (s' := [N_nuc]) ztab Zc
  let wd := paramBlock OFF_WD [N_nuc, 3, F] θ
  let ws := paramBlock OFF_WS [N_nuc, F] θ
  let wz := paramBlock OFF_WZ [N_nuc, F] θ
  let dispT := Xla.transpose disp perm[1, 2, 0]
  let distT := Xla.transpose dist perm[1, 0]
  let zB := Xla.broadcast [⟨N, false⟩, ⟨N_nuc, true⟩] zRaw
  let zT := Xla.transpose zB perm[1, 0]
  let tD := Xla.einsum (s := [N_nuc, 3, N, F]) [[#0, #1, #2], [#0, #1, #3]] 2
    (dispT.append wd)
  let tS := Xla.einsum (s := [N_nuc, N, F]) [[#0, #1], [#0, #2]] 1 (distT.append ws)
  let tZ := Xla.einsum (s := [N_nuc, N, F]) [[#0, #1], [#0, #2]] 1 (zT.append wz)
  let b0 := Xla.broadcast [⟨N, false⟩, ⟨F, true⟩] (paramSlice OFF_B0 F θ)
  tanh (Xla.add (Xla.add (Xla.add tD tS) tZ) b0)

/-- Initial same-spin two-electron stream: per-pair displacement and distance
features contracted to width `F`, then `tanh`. -/
def twoStreamInit (N : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N, F]⟩] :=
  let pd := pairDisp N r
  let pd2 := pairDist N r θ
  let we := paramBlock OFF_WE [3, F] θ
  let wd := paramSlice OFF_WD2 F θ
  let pdT := Xla.transpose pd perm[2, 0, 1]
  let tD := Xla.einsum (s := [3, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (pdT.append we)
  let wdB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] wd
  let pd2B := Xla.broadcast [⟨N, true⟩, ⟨N, true⟩, ⟨F, false⟩] pd2
  let tS := Xla.mul pd2B wdB
  let gb := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] (paramSlice OFF_GB F θ)
  tanh (Xla.add (Xla.add tD tS) gb)

/-- Initial opposite-spin two-electron stream. -/
def twoStreamInitCross (N N' : ℕ) (r : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (r' : Expr XlaOp args [⟨.float, [N', 3]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N', F]⟩] :=
  let pd := pairDispCross N N' r r'
  let pd2 := pairDistCross N N' r r' θ
  let we := paramBlock OFF_WE [3, F] θ
  let wd := paramSlice OFF_WD2 F θ
  let pdT := Xla.transpose pd perm[2, 0, 1]
  let tD := Xla.einsum (s := [3, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (pdT.append we)
  let wdB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] wd
  let pd2B := Xla.broadcast [⟨N, true⟩, ⟨N', true⟩, ⟨F, false⟩] pd2
  let tS := Xla.mul pd2B wdB
  let gb := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] (paramSlice OFF_GB F θ)
  tanh (Xla.add (Xla.add tD tS) gb)

/-! ### Layer updates -/

/-- One-electron stream update: `h ← tanh(V h + Σⱼ w ⊙ g_ij + Σⱼ w' ⊙ g_ij^{σσ̄} + b)`. -/
def oneStreamLayer (N N' : ℕ) (h : Expr XlaOp args [⟨.float, [N, F]⟩])
    (g : Expr XlaOp args [⟨.float, [N, N, F]⟩])
    (gC : Expr XlaOp args [⟨.float, [N, N', F]⟩]) (off : ℕ)
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, F]⟩] :=
  let V := paramBlock off [F, F] θ
  let w := paramSlice (off + 1024) F θ
  let wC := paramSlice (off + 1056) F θ
  let b := paramSlice (off + 1088) F θ
  let tV := Xla.einsum (s := [F, N, F]) [[#1, #0], [#0, #2]] 1 (h.append V)
  let tG := Xla.einsum (s := [N, N, F]) [[#0, #1, #2], [#2]] 1 (g.append w)
  let gCT := Xla.transpose gC perm[1, 0, 2]
  let tGC := Xla.einsum (s := [N', N, F]) [[#0, #1, #2], [#2]] 1 (gCT.append wC)
  let bB := Xla.broadcast [⟨N, false⟩, ⟨F, true⟩] b
  tanh (Xla.add (Xla.add (Xla.add tV tG) tGC) bB)

/-- Same-spin two-electron stream update:
`g ← tanh(G g + H (h_i + h_j) + c)`. -/
def twoStreamLayer (N : ℕ) (g : Expr XlaOp args [⟨.float, [N, N, F]⟩])
    (h : Expr XlaOp args [⟨.float, [N, F]⟩]) (off : ℕ)
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N, F]⟩] :=
  let G := paramBlock (off + 1120) [F, F] θ
  let H := paramBlock (off + 2144) [F, F] θ
  let c := paramSlice (off + 3168) F θ
  let gT := Xla.transpose g perm[2, 0, 1]
  let tG := Xla.einsum (s := [F, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (gT.append G)
  let hA := Xla.broadcast [⟨N, false⟩, ⟨N, true⟩, ⟨F, true⟩] h
  let hB := Xla.broadcast [⟨N, true⟩, ⟨N, false⟩, ⟨F, true⟩] h
  let hS := Xla.transpose (Xla.add hA hB) perm[2, 0, 1]
  let tH := Xla.einsum (s := [F, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (hS.append H)
  let cB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] c
  tanh (Xla.add (Xla.add tG tH) cB)

/-- Opposite-spin two-electron stream update:
`g^{σσ̄} ← tanh(G' g^{σσ̄} + H' (h_i^σ + h_j^{σ̄}) + c')`. -/
def twoStreamLayerCross (N N' : ℕ) (gC : Expr XlaOp args [⟨.float, [N, N', F]⟩])
    (h : Expr XlaOp args [⟨.float, [N, F]⟩])
    (h' : Expr XlaOp args [⟨.float, [N', F]⟩]) (off : ℕ)
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [N, N', F]⟩] :=
  let G := paramBlock (off + 3200) [F, F] θ
  let H := paramBlock (off + 4224) [F, F] θ
  let c := paramSlice (off + 5248) F θ
  let gT := Xla.transpose gC perm[2, 0, 1]
  let tG := Xla.einsum (s := [F, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (gT.append G)
  let hA := Xla.broadcast [⟨N, true⟩, ⟨N', false⟩, ⟨F, true⟩] h
  let hB := Xla.broadcast [⟨N, false⟩, ⟨N', true⟩, ⟨F, true⟩] h'
  let hS := Xla.transpose (Xla.add hA hB) perm[2, 0, 1]
  let tH := Xla.einsum (s := [F, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (hS.append H)
  let cB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] c
  tanh (Xla.add (Xla.add tG tH) cB)

/-! ### Outputs -/

/-- Bounded Jastrow factor `J = Σᵢ w_J · hᵢ` (scalar). -/
def jastrow (N : ℕ) (h : Expr XlaOp args [⟨.float, [N, F]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, []⟩] :=
  let wJ := paramSlice OFF_WJ F θ
  Xla.einsum (s := [N, F]) [[#0, #1], [#1]] 2 (h.append wJ)

/-- Per-orbital exponential envelope, shape `[K, N, N]`: entry `(k, j, i)` is
`exp(-Σα exp A[k,i,α] · ‖xⱼ - Rα‖)`, the FermiNet envelope with decay exponents
parameterized by *orbital* `(k, i)` and evaluated on electron `j` (FermiNet,
Eq. 19). Indexing the parameters by orbital — not by electron — is what makes the
envelope a permutation-*equivariant* function of the electron positions. -/
def envelope (N N_nuc : ℕ) (dist : Expr XlaOp args [⟨.float, [N, N_nuc]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [K, N, N]⟩] :=
  let A := paramBlock OFF_A [K, N, N_nuc] θ
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
def orbitalMatrix (N : ℕ) (h : Expr XlaOp args [⟨.float, [N, F]⟩])
    (envExp : Expr XlaOp args [⟨.float, [K, N, N]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, [K, N, N]⟩] :=
  let worb := paramBlock OFF_WORB [F, K, N] θ
  let φ := Xla.einsum (s := [F, N, K, N]) [[#1, #0], [#0, #2, #3]] 1 (h.append worb)
  let φT := Xla.transpose φ perm[1, 0, 2]
  Xla.mul φT envExp

/-- `det` as a library function, for `vmap`-ing over the determinant axis. -/
def detExpr (n : ℕ) : Expr XlaOp [⟨.float, [n, n]⟩] [⟨.float, []⟩] :=
  Soir.Expr.ofFn fun x => Xla.det x

/-- Weighted sum of the `K` determinants of a spin sector: `Σₖ exp w_k · detₖ`. -/
def detBlock (N : ℕ) (orb : Expr XlaOp args [⟨.float, [K, N, N]⟩])
    (θ : Expr XlaOp args [⟨.float, [N_PARAM]⟩]) :
    Expr XlaOp args [⟨.float, []⟩] :=
  let dets := Xla.vmap (batch := K) (ins := [⟨.float, [N, N]⟩]) (outs := [⟨.float, []⟩])
    (detExpr N) orb .nil
  let wdet := Xla.exp (paramSlice OFF_WDET K θ)
  Xla.sum 1 (Xla.mul dets wdet)

end

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
  Soir.Expr.ofFn fun θ => fun R => fun Z => fun rUp => fun rDown =>
    let hU0 := oneStreamInit N_up N_nuc rUp R Z θ
    let hD0 := oneStreamInit N_down N_nuc rDown R Z θ
    let gU0 := twoStreamInit N_up rUp θ
    let gD0 := twoStreamInit N_down rDown θ
    let gUD0 := twoStreamInitCross N_up N_down rUp rDown θ
    let gDU0 := twoStreamInitCross N_down N_up rDown rUp θ
    -- layer 0
    let hU1 := oneStreamLayer N_up N_down hU0 gU0 gUD0 OFF_L0 θ
    let hD1 := oneStreamLayer N_down N_up hD0 gD0 gDU0 OFF_L0 θ
    let gU1 := twoStreamLayer N_up gU0 hU0 OFF_L0 θ
    let gD1 := twoStreamLayer N_down gD0 hD0 OFF_L0 θ
    let gUD1 := twoStreamLayerCross N_up N_down gUD0 hU0 hD0 OFF_L0 θ
    let gDU1 := twoStreamLayerCross N_down N_up gDU0 hD0 hU0 OFF_L0 θ
    -- layer 1
    let hU2 := oneStreamLayer N_up N_down hU1 gU1 gUD1 OFF_L1 θ
    let hD2 := oneStreamLayer N_down N_up hD1 gD1 gDU1 OFF_L1 θ
    -- Jastrow factor (the streams are tanh-bounded, so `exp J` is bounded)
    let JU := jastrow N_up hU2 θ
    let JD := jastrow N_down hD2 θ
    let J := Xla.exp (Xla.add JU JD)
    -- exponential envelopes
    let distU := enDist N_up N_nuc rUp R θ
    let distD := enDist N_down N_nuc rDown R θ
    let envU := Xla.exp (Xla.neg (envelope N_up N_nuc distU θ))
    let envD := Xla.exp (Xla.neg (envelope N_down N_nuc distD θ))
    -- determinants
    let orbU := orbitalMatrix N_up hU2 envU θ
    let orbD := orbitalMatrix N_down hD2 envD θ
    let detU := detBlock N_up orbU θ
    let detD := detBlock N_down orbD θ
    Xla.mul (Xla.mul J detU) detD

/-- The FermiNet `Ansatz`. -/
def fermiNet : Ansatz where
  N_param := N_PARAM
  ansatz := fun N_nuc N_up N_down => fermiNetAnsatz N_nuc N_up N_down

/-! ### The validity contract (all proofs `sorry`)

See the module docstring for the proof sketches. -/

/-- Antisymmetry in both spin sectors: the streams are permutation-equivariant,
the envelope/Jastrow invariant, and each determinant picks up `sign σ`. -/
theorem antisymmetric (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNet.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    IsAntisymmetric ((fermiNet.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc) := by
  sorry

/-- Smoothness: every primitive used is real-analytic on its domain (`sqrt` guarded
away from 0 by `ε > 0`, `tanh` a composition of analytic functions), so `Ψ` is C². -/
theorem smooth (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNet.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    ContDiff ℝ 2 (Function.uncurry ((fermiNet.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc)) := by
  sorry

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
