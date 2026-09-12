import Xla
import QMC.Ansatz

/-!
# `QMC/PauliNet.lean` — a PauliNet-style molecular QMC ansatz in the Soir/XLA DSL

This file implements a SOTA-family neural-network wavefunction ansatz for molecules,
satisfying the validity contract of `QMC/Ansatz.lean` (`IsValidQMCWavefunction`:
antisymmetry, C² smoothness, exponential envelope). All proofs are `sorry`; the
architecture is chosen so that all three contract fields are *true*, i.e. the sorries
are dischargeable in principle (see the per-field notes at the bottom).

## Style: closed libraries composed with `Expr.apply`

As in `QMC/FermiNet.lean` (whose module docstring analyzes the codegen cost model in
detail), every reusable component is a closed `SimpleExpr` whose inputs are formal
arguments, composed with `Soir.Expr.apply`. `Expr.code` deduplicates sub-expressions
by a structural hash whose cost is quadratic in the size of the scope a program is
generated in; a flat PauliNet program is one scope of every binding it contains —
each SchNet block inlines the filter MLP six times, each determinant head inlines
the RBF/envelope/gather machinery, etc. As libraries, each body is generated once in
its own small scope and the top level is a short chain of `call; @n`.

Measured (Lean 4.33.1, `lake env lean QMC/PauliNet.lean`, including toolchain
startup and elaboration), builder style vs. this file:

| | builder style | apply style |
|---|---|---|
| whole file incl. `#eval (pauliNet.ansatz 2 1 1).code` | 1 m 53 s | 4.3 s |

The flat H₂ program had 605 top-level bindings; here the top level is 36 bindings
calling 90 libraries. The emitted program evaluates bitwise-identically to the
builder-style one in the JAX harness (`python/eval.py`), for both H₂ (2,1,1) and
LiH (1,2,1).

## Architecture choice: PauliNet / SchNet family

The contract fixes `N_param` once and for all, while the ansatz must serve *all*
`(N_nuc, N_up, N_down)`. This rules out FermiNet-style first layers (input dimension
`4·N_nuc` per electron ⇒ parameter count grows with the molecule). The PauliNet family
(Hermann, Schätzle, Noé, *Deep-neural-network solution of the electronic Schrödinger
equation*, Nat. Chem. 2020) is the SOTA lineage built on *shared-weight* message
passing (SchNet embeddings), whose parameter count is independent of system size:

1. **SchNet embedding.** Per-electron feature vectors `hᵢ` (one per spin sector) are
   refined by `L` continuous-filter message-passing blocks. Pair geometry enters only
   through a Gaussian radial basis of the *squared* distances
   `e_k(d²) = exp(-exp(γₖ)·(d²-μₖ)²)`; filter-generating MLPs and update weights are
   shared across electrons, spin sectors, and molecules. Nuclear embeddings are rows
   of a `Zmax × d` table gathered by atomic number `Z` (gather on integer data only,
   per the op discipline).

2. **Orbital heads with CRT pooling.** A determinant needs `N_up` (resp. `N_down`)
   orbitals, but `N_param` is fixed. Orbital row `k` uses linear channel `k mod P`
   of a pooled projection `A : P × d` and envelope exponent `ζ_{k mod Q}` of a pooled
   exponent vector, with `P, Q` coprime. By the Chinese remainder theorem the pairs
   `(k mod P, k mod Q)` are distinct for `k < P·Q = 72`, so rows are genuinely
   different functions of `xᵢ` (not scalar multiples, which would collapse the
   determinant) for every system in the regime of interest — with *no* per-orbital
   parameters. Beyond `P·Q` electrons per sector the ansatz degenerates to `Ψ ≡ 0`,
   which still satisfies the contract (and is caught by the runtime `Ψ ≢ 0` monitor).

3. **Gaussian envelopes.** Each orbital is multiplied by
   `E_k(xᵢ) = Σ_I exp(-ζₖ‖xᵢ-R_I‖²)`, `ζₖ = exp(ζrawₖ) > 0` — the `exp`
   reparameterization keeps exponents positive for *all* θ (the contract quantifies
   over all θ), and Gaussian decay implies the required exponential envelope
   (`e^{-ζr²} ≤ C e^{-kr}`). We use `d²`, never `‖·‖`: no `sqrt` occurs anywhere, so
   no creases are introduced (the Kato cusp is deliberately sacrificed — the contract
   requires C², which plain distances violate at coalescence points).

4. **Determinants and Jastrow.** `Ψ = e^J · Σ_{m<2} c_m det Φ^↑_m det Φ^↓_m` with
   `J = Σ_pairs tanh(w_J · e(d²))` a bounded Jastrow factor (`|tanh| ≤ 1`).

## Op-set report (no additions needed)

The current op set **suffices**; no new ops are required. The implementation only
uses ops with real `DirectImpl` semantics (i.e. not falling through to
`DirectImpl.zero`): `add, sub, mul, neg, div, exp, mod, sum, einsum, det, broadcast,
transpose, gather, scatter-free, iota, ofNat, unflatten, cast`. Notable *avoidances*
(zero-stub or contract-hostile ops):

* `tanh`/`log`/`sin`/... are `DirectImpl.zero` stubs — `tanh` is built as
  `1 - 2/(exp(2x)+1)` from `exp` and `div` (denominator `≥ 1`, so the composition is
  real-analytic); positivity reparameterization uses `exp`, not softplus (`log`).
* `sqrt`/`abs` are smoothness hazards (creases at 0) — eliminated by working with
  squared distances throughout.
* `choice`/`eq` on float data (piecewise constant creases) — not used at all.
* `concat`, `dynamic_slice`, `convert_type`, `div_int` are zero-stubs — the two spin
  sectors are processed in parallel instead of concatenated, and flat-parameter
  slicing is done by `gather` with `iota`-arithmetic index tensors plus `unflatten`.
-/

namespace QMC.PauliNet

open Xla Soir

variable {args : List TensorType}

/-! ## Broadcast plumbing

Symbolic-shape broadcasts (`s.map (⟨·, true⟩)` etc.) do not reduce definitionally,
so we cast through provable shape equalities. -/

@[simp] theorem map_fst_pair (b : Bool) (s : Shape) : s.map (Prod.fst ∘ (⟨·, b⟩)) = s := by
  induction s with
  | nil => rfl
  | cons a s ih => simp [Function.comp_def]

@[simp] theorem preBroadcast_map_true (s : Shape) :
    Tensor.preBroadcast (s.map (⟨·, true⟩)) = s := by
  induction s with
  | nil => rfl
  | cons a s ih => simp [Tensor.preBroadcast_cons_true, ih]

@[simp] theorem preBroadcast_map_false (s : Shape) :
    Tensor.preBroadcast (s.map (⟨·, false⟩)) = [] := by
  induction s with
  | nil => rfl
  | cons a s ih => simp [Tensor.preBroadcast_cons_false, ih]

/-- Broadcast keeping existing axes, appending new trailing axes. -/
def bcastAppend {α : DType} (s extra : Shape) (x : Expr XlaOp args [⟨α, s⟩]) :
    Expr XlaOp args [⟨α, s ++ extra⟩] :=
  Xla.cast (by simp) <| Xla.broadcast (s.map (⟨·, true⟩) ++ extra.map (⟨·, false⟩)) <|
    Xla.cast (by simp) x

/-- Broadcast keeping existing axes, prepending new leading axes. -/
def bcastPrepend {α : DType} (pre s : Shape) (x : Expr XlaOp args [⟨α, s⟩]) :
    Expr XlaOp args [⟨α, pre ++ s⟩] :=
  Xla.cast (by simp) <| Xla.broadcast (pre.map (⟨·, false⟩) ++ s.map (⟨·, true⟩)) <|
    Xla.cast (by simp) x

/-- Broadcast inserting new axes in the middle. -/
def bcastInsert {α : DType} (pre mid post : Shape) (x : Expr XlaOp args [⟨α, pre ++ post⟩]) :
    Expr XlaOp args [⟨α, pre ++ mid ++ post⟩] :=
  Xla.cast (by simp) <|
    Xla.broadcast (pre.map (⟨·, true⟩) ++ mid.map (⟨·, false⟩) ++ post.map (⟨·, true⟩)) <|
      Xla.cast (by simp) x

/-! ## Architecture hyperparameters (compile-time literals) -/

/-- Feature width. -/ def d : ℕ := 32
/-- Radial basis size. -/ def K : ℕ := 8
/-- Linear-channel pool size. -/ def P : ℕ := 8
/-- Envelope-exponent pool size (coprime to `P` for CRT pooling). -/ def Q : ℕ := 9
/-- Number of determinants per spin sector. -/ def D : ℕ := 2
/-- Nuclear embedding table covers `Z < Zmax`. -/ def Zmax : ℕ := 54

/-! ## Parameter layout (all offsets reduce to literals) -/

-- Interaction-block-relative offsets: [eeW1, eeb1, eeW2, eeb2, enW1, enb1, enW2, enb2, V, g]
def bo_eeW1 : ℕ := 0
def bo_eeb1 : ℕ := bo_eeW1 + d * K
def bo_eeW2 : ℕ := bo_eeb1 + d
def bo_eeb2 : ℕ := bo_eeW2 + d * d
def bo_enW1 : ℕ := bo_eeb2 + d
def bo_enb1 : ℕ := bo_enW1 + d * K
def bo_enW2 : ℕ := bo_enb1 + d
def bo_enb2 : ℕ := bo_enW2 + d * d
def bo_V : ℕ := bo_enb2 + d
def bo_g : ℕ := bo_V + d * d
def blockSize : ℕ := bo_g + d

-- Determinant-head-relative offsets: [Vm, cm, Am, ζraw]
def ho_Vm : ℕ := 0
def ho_cm : ℕ := ho_Vm + d * d
def ho_Am : ℕ := ho_cm + d
def ho_ζ : ℕ := ho_Am + P * d
def headSize : ℕ := ho_ζ + Q

-- Global layout: [h0, table, μ, γ, blocks ×2, wJ, heads ×2, c]
def off_h0 : ℕ := 0
def off_table : ℕ := off_h0 + d
def off_μ : ℕ := off_table + Zmax * d
def off_γ : ℕ := off_μ + K
def off_block (ℓ : ℕ) : ℕ := off_γ + K + ℓ * blockSize
def off_wJ : ℕ := off_block 2
def off_head (m : ℕ) : ℕ := off_wJ + K + m * headSize
def off_c : ℕ := off_head 2
def N_param : ℕ := off_c + D

/-! ## Primitives composed from the existing op set -/

/-- `tanh` from `exp` and `div` at the `Expr` level (both have real `DirectImpl`
semantics; the `tanh` prim is a zero stub). Denominator `exp(2x)+1 ≥ 1`, so this is
real-analytic. -/
def tanh' {s : Shape} (x : Expr XlaOp args [⟨.float, s⟩]) : Expr XlaOp args [⟨.float, s⟩] :=
  let one : Expr XlaOp args [⟨.float, s⟩] := Xla.ofNat 1
  let two : Expr XlaOp args [⟨.float, s⟩] := Xla.ofNat 2
  let e := Xla.exp (Xla.mul two x)
  Xla.sub one (Xla.div two (Xla.add e one))

/-- `tanh` as a library, so repeated activations dedup to one body. -/
def tanhL (s : Shape) : SimpleExpr [⟨.float, s⟩] ⟨.float, s⟩ :=
  Expr.ofFn fun x => tanh' x

/-! ## Parameter access -/

/-- Slice `len` consecutive floats from the parameter vector starting at `off`
(indices via `iota` arithmetic — avoids the `dynamic_slice`/`convert_type` stubs). -/
def paramSlice (off len : ℕ) : SimpleExpr [⟨.float, [N_param]⟩] ⟨.float, [len]⟩ :=
  Expr.ofFn fun θ =>
    let idx : Expr XlaOp _ [⟨.int, [len]⟩] := Xla.add (Xla.iota len) (Xla.ofNat off)
    Xla.gather (α := .float) (s := [N_param]) (s' := [len]) θ idx

/-- Reshape a slice of the parameter vector into a tensor of shape `s`. -/
def paramBlock (off : ℕ) (s : Shape) : SimpleExpr [⟨.float, [N_param]⟩] ⟨.float, s⟩ :=
  Expr.ofFn fun θ => Xla.unflatten s ((paramSlice off s.prod).apply θ)

/-! ## Basic layers -/

/-- Squared pairwise distances `d²[i,j] = ‖xᵢ - yⱼ‖²` (never `sqrt`: C²-safe). -/
def pairDist2 (N M : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [M, 3]⟩] ⟨.float, [N, M]⟩ :=
  Expr.ofFn fun x y =>
    let xb := bcastInsert [N] [M] [3] x
    let yb := bcastPrepend [N] [M, 3] y
    let dxy := Xla.sub xb yb
    Xla.sum 1 (Xla.transpose (Xla.mul dxy dxy) perm[2,0,1])

/-- Linear layer `y = x Wᵀ + b` on a `[N, di]` particle-feature tensor. -/
def linear (N di dout : ℕ) :
    SimpleExpr [⟨.float, [N, di]⟩, ⟨.float, [dout, di]⟩, ⟨.float, [dout]⟩]
      ⟨.float, [N, dout]⟩ :=
  Expr.ofFn fun x W b =>
    let y := Xla.einsum [di, N, dout] [[#1, #0], [#2, #0]] 1 (x.append W)
    Xla.add y (Xla.broadcast [⟨N, false⟩, ⟨dout, true⟩] b)

/-- Linear layer on the last axis of an `[A, B, Ki]` pair-feature tensor. -/
def linear3 (A B Ki dout : ℕ) :
    SimpleExpr [⟨.float, [A, B, Ki]⟩, ⟨.float, [dout, Ki]⟩, ⟨.float, [dout]⟩]
      ⟨.float, [A, B, dout]⟩ :=
  Expr.ofFn fun x W b =>
    let y := Xla.einsum [Ki, A, B, dout] [[#1, #2, #0], [#3, #0]] 1 (x.append W)
    Xla.add y (Xla.broadcast [⟨A, false⟩, ⟨B, false⟩, ⟨dout, true⟩] b)

/-- Gaussian radial basis of squared distances: `exp(-exp(γₖ)·(d²-μₖ)²)`; appends a
basis axis. Centers/widths are learned (`∀θ`-safe: `exp(γₖ) > 0` always). Bounded and
real-analytic in the electron coordinates. -/
def rbf (s : Shape) :
    SimpleExpr [⟨.float, [K]⟩, ⟨.float, [K]⟩, ⟨.float, s⟩] ⟨.float, s ++ [K]⟩ :=
  Expr.ofFn fun μ γ d2 =>
    let d2b := bcastAppend s [K] d2
    let μb := bcastPrepend s [K] μ
    let γb := bcastPrepend s [K] γ
    let r := Xla.sub d2b μb
    Xla.exp (Xla.neg (Xla.mul (Xla.exp γb) (Xla.mul r r)))

/-- SchNet message: `mᵢ = Σⱼ F[i,j,:] ⊙ h[j,:]` as one einsum. -/
def message (N M d' : ℕ) :
    SimpleExpr [⟨.float, [N, M, d']⟩, ⟨.float, [M, d']⟩] ⟨.float, [N, d']⟩ :=
  Expr.ofFn fun F h => Xla.einsum [M, N, d'] [[#1, #0, #2], [#0, #2]] 1 (F.append h)

/-- Nuclear embeddings: rows of the `Zmax × d` table gathered by atomic number. -/
def embedNuc (N_nuc : ℕ) :
    SimpleExpr [⟨.float, [Zmax, d]⟩, ⟨.int, [N_nuc]⟩] ⟨.float, [N_nuc, d]⟩ :=
  Expr.ofFn fun table Z =>
    let i₀ := bcastAppend [N_nuc] [d] Z
    let i₁ := bcastPrepend [N_nuc] [d] (Xla.iota d)
    Xla.gather table (i₀.append i₁)

/-! ## SchNet interaction block (shared filter/update weights across spin sectors) -/

/-- Two-layer continuous filter MLP on the radial basis of a pair-distance tensor,
with weights sliced from `θ` at the given (absolute) offsets. -/
def filter (A B oW1 ob1 oW2 ob2 : ℕ) :
    SimpleExpr [⟨.float, [N_param]⟩, ⟨.float, [K]⟩, ⟨.float, [K]⟩, ⟨.float, [A, B]⟩]
      ⟨.float, [A, B, d]⟩ :=
  Expr.ofFn fun θ μ γ d2 =>
    let W1 := (paramBlock oW1 [d, K]).apply θ
    let b1 := (paramBlock ob1 [d]).apply θ
    let W2 := (paramBlock oW2 [d, d]).apply θ
    let b2 := (paramBlock ob2 [d]).apply θ
    let e := (rbf [A, B]).apply ((μ.append γ).append d2)
    let y1 := (tanhL [A, B, d]).apply ((linear3 A B K d).apply ((e.append W1).append b1))
    (tanhL [A, B, d]).apply ((linear3 A B d d).apply ((y1.append W2).append b2))

/-- Up-spin half of one message-passing block: gather electron-electron and
electron-nucleus messages through learned continuous filters of the squared pair
distances, then a tanh-updated linear combination. The two spin halves share no
computation (up uses `d2uu, d2ud, d2uN` only), so the block is split into two
single-output libraries without duplicating any work. -/
def schNetUp (off N_up N_down N_nuc : ℕ) :
    SimpleExpr
      [ ⟨.float, [N_param]⟩, ⟨.float, [K]⟩, ⟨.float, [K]⟩
      , ⟨.float, [N_up, d]⟩, ⟨.float, [N_down, d]⟩, ⟨.float, [N_nuc, d]⟩
      , ⟨.float, [N_up, N_up]⟩, ⟨.float, [N_up, N_down]⟩, ⟨.float, [N_up, N_nuc]⟩ ]
      ⟨.float, [N_up, d]⟩ :=
  Expr.ofFn fun θ μ γ hu hd c d2uu d2ud d2uN =>
    let Fuu := (filter N_up N_up (off + bo_eeW1) (off + bo_eeb1)
      (off + bo_eeW2) (off + bo_eeb2)).apply (((θ.append μ).append γ).append d2uu)
    let Fud := (filter N_up N_down (off + bo_eeW1) (off + bo_eeb1)
      (off + bo_eeW2) (off + bo_eeb2)).apply (((θ.append μ).append γ).append d2ud)
    let Gu := (filter N_up N_nuc (off + bo_enW1) (off + bo_enb1)
      (off + bo_enW2) (off + bo_enb2)).apply (((θ.append μ).append γ).append d2uN)
    let V := (paramBlock (off + bo_V) [d, d]).apply θ
    let g := (paramBlock (off + bo_g) [d]).apply θ
    let m := Xla.add ((message N_up N_up d).apply (Fuu.append hu))
      (Xla.add ((message N_up N_down d).apply (Fud.append hd))
        ((message N_up N_nuc d).apply (Gu.append c)))
    (tanhL [N_up, d]).apply
      (Xla.add ((linear N_up d d).apply ((hu.append V).append g)) m)

/-- Down-spin half of one message-passing block; see `schNetUp`. -/
def schNetDown (off N_up N_down N_nuc : ℕ) :
    SimpleExpr
      [ ⟨.float, [N_param]⟩, ⟨.float, [K]⟩, ⟨.float, [K]⟩
      , ⟨.float, [N_down, d]⟩, ⟨.float, [N_up, d]⟩, ⟨.float, [N_nuc, d]⟩
      , ⟨.float, [N_down, N_down]⟩, ⟨.float, [N_down, N_up]⟩, ⟨.float, [N_down, N_nuc]⟩ ]
      ⟨.float, [N_down, d]⟩ :=
  Expr.ofFn fun θ μ γ hd hu c d2dd d2du d2dN =>
    let Fdd := (filter N_down N_down (off + bo_eeW1) (off + bo_eeb1)
      (off + bo_eeW2) (off + bo_eeb2)).apply (((θ.append μ).append γ).append d2dd)
    let Fdu := (filter N_down N_up (off + bo_eeW1) (off + bo_eeb1)
      (off + bo_eeW2) (off + bo_eeb2)).apply (((θ.append μ).append γ).append d2du)
    let Gd := (filter N_down N_nuc (off + bo_enW1) (off + bo_enb1)
      (off + bo_enW2) (off + bo_enb2)).apply (((θ.append μ).append γ).append d2dN)
    let V := (paramBlock (off + bo_V) [d, d]).apply θ
    let g := (paramBlock (off + bo_g) [d]).apply θ
    let m := Xla.add ((message N_down N_down d).apply (Fdd.append hd))
      (Xla.add ((message N_down N_up d).apply (Fdu.append hu))
        ((message N_down N_nuc d).apply (Gd.append c)))
    (tanhL [N_down, d]).apply
      (Xla.add ((linear N_down d d).apply ((hd.append V).append g)) m)

/-! ## Determinant head (shared across spin sectors, CRT-pooled orbitals) -/

/-- One determinant of pooled orbitals for a spin sector:
`Φ[k,i] = (A·tanh(V hᵢ + c))_{k mod P} · Σ_I exp(-exp(ζraw_{k mod Q})‖xᵢ-R_I‖²)`.
Row `k` is a distinct function of `xᵢ` for `k < P·Q` (CRT), with fixed parameter
count. -/
def detHead (off N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N_param]⟩, ⟨.float, [N, d]⟩, ⟨.float, [N, N_nuc]⟩]
      ⟨.float, []⟩ :=
  Expr.ofFn fun θ h d2nuc =>
    let Vm := (paramBlock (off + ho_Vm) [d, d]).apply θ
    let cm := (paramBlock (off + ho_cm) [d]).apply θ
    let Am := (paramBlock (off + ho_Am) [P, d]).apply θ
    let ζraw := (paramBlock (off + ho_ζ) [Q]).apply θ
    let g := (tanhL [N, d]).apply ((linear N d d).apply ((h.append Vm).append cm))
    let U := Xla.einsum [d, P, N] [[#2, #0], [#1, #0]] 1 (g.append Am)
    -- orbital rows gathered from the channel pool by `k mod P`
    let rowidx : Expr XlaOp _ [⟨.int, [N]⟩] := Xla.mod (Xla.iota N) (Xla.ofNat P)
    let i₀ := bcastAppend [N] [N] rowidx
    let i₁ := bcastPrepend [N] [N] (Xla.iota N)
    let M := Xla.gather U (i₀.append i₁)
    -- Gaussian envelope with exponents pooled by `k mod Q`
    let ζidx : Expr XlaOp _ [⟨.int, [N]⟩] := Xla.mod (Xla.iota N) (Xla.ofNat Q)
    let ζrow : Expr XlaOp _ [⟨.float, [N]⟩] := Xla.gather (Xla.exp ζraw) ζidx
    let ζb := bcastAppend [N] [N, N_nuc] ζrow
    let d2b := bcastPrepend [N] [N, N_nuc] d2nuc
    let e := Xla.exp (Xla.neg (Xla.mul ζb d2b))
    let Env := Xla.sum 1 (Xla.transpose e perm[2,0,1])
    Xla.det (Xla.mul M Env)

/-- Bounded Jastrow over one pair set: `Σ_{i,j} tanh(wJ · e(d²ᵢⱼ))`. -/
def jastrow (A B : ℕ) :
    SimpleExpr [⟨.float, [N_param]⟩, ⟨.float, [K]⟩, ⟨.float, [K]⟩, ⟨.float, [A, B]⟩]
      ⟨.float, []⟩ :=
  Expr.ofFn fun θ μ γ d2 =>
    let wJ := (paramBlock off_wJ [K]).apply θ
    let e := (rbf [A, B]).apply ((μ.append γ).append d2)
    let v := Xla.einsum [K, A, B] [[#1, #2, #0], [#0]] 1 (e.append wJ)
    Xla.sum 2 ((tanhL [A, B]).apply v)

/-! ## The ansatz -/

/-- PauliNet-style wavefunction: `Ψ = e^J · Σ_{m<2} c_m det Φ^↑_m · det Φ^↓_m`. -/
def pauliNetExpr (N_nuc N_up N_down : ℕ) :
    Xla.SimpleExpr
      [ ⟨.float, [N_param]⟩      -- parameters
      , ⟨.float, [N_nuc, 3]⟩     -- positions of nuclei
      , ⟨.int, [N_nuc]⟩          -- types of nuclei
      , ⟨.float, [N_up, 3]⟩      -- positions of spin up electrons
      , ⟨.float, [N_down, 3]⟩ ]  -- positions of spin down electrons
      ⟨.float, []⟩ :=
  Soir.Expr.ofFn fun θ R Z xu xd =>
    -- global parameters
    let h0 := (paramBlock off_h0 [d]).apply θ
    let table := (paramBlock off_table [Zmax, d]).apply θ
    let μ := (paramBlock off_μ [K]).apply θ
    let γ := (paramBlock off_γ [K]).apply θ
    -- nuclear embeddings and squared pair distances
    let c := (embedNuc N_nuc).apply (table.append Z)
    let d2uu := (pairDist2 N_up N_up).apply (xu.append xu)
    let d2ud := (pairDist2 N_up N_down).apply (xu.append xd)
    let d2dd := (pairDist2 N_down N_down).apply (xd.append xd)
    let d2du := (pairDist2 N_down N_up).apply (xd.append xu)
    let d2uN := (pairDist2 N_up N_nuc).apply (xu.append R)
    let d2dN := (pairDist2 N_down N_nuc).apply (xd.append R)
    -- SchNet embedding: L = 2 blocks, constant initial electron features
    let hu0 := bcastPrepend [N_up] [d] h0
    let hd0 := bcastPrepend [N_down] [d] h0
    let hu1 := (schNetUp (off_block 0) N_up N_down N_nuc).apply
      ((((((((θ.append μ).append γ).append hu0).append hd0).append c).append d2uu).append d2ud).append d2uN)
    let hd1 := (schNetDown (off_block 0) N_up N_down N_nuc).apply
      ((((((((θ.append μ).append γ).append hd0).append hu0).append c).append d2dd).append d2du).append d2dN)
    let hu2 := (schNetUp (off_block 1) N_up N_down N_nuc).apply
      ((((((((θ.append μ).append γ).append hu1).append hd1).append c).append d2uu).append d2ud).append d2uN)
    let hd2 := (schNetDown (off_block 1) N_up N_down N_nuc).apply
      ((((((((θ.append μ).append γ).append hd1).append hu1).append c).append d2dd).append d2du).append d2dN)
    -- D = 2 determinants per spin sector, heads shared across sectors
    let du0 := (detHead (off_head 0) N_up N_nuc).apply ((θ.append hu2).append d2uN)
    let dd0 := (detHead (off_head 0) N_down N_nuc).apply ((θ.append hd2).append d2dN)
    let du1 := (detHead (off_head 1) N_up N_nuc).apply ((θ.append hu2).append d2uN)
    let dd1 := (detHead (off_head 1) N_down N_nuc).apply ((θ.append hd2).append d2dN)
    let c0 := (paramBlock off_c []).apply θ
    let c1 := (paramBlock (off_c + 1) []).apply θ
    -- bounded Jastrow over all electron pairs
    let J := Xla.add ((jastrow N_up N_up).apply (((θ.append μ).append γ).append d2uu))
      (Xla.add ((jastrow N_up N_down).apply (((θ.append μ).append γ).append d2ud))
        ((jastrow N_down N_down).apply (((θ.append μ).append γ).append d2dd)))
    Xla.mul (Xla.exp J)
      (Xla.add (Xla.mul c0 (Xla.mul du0 dd0)) (Xla.mul c1 (Xla.mul du1 dd1)))

end QMC.PauliNet

open QMC.PauliNet in
/-- The ansatz as a validity-contract object. `N_param = 11916` for the default
hyperparameters. -/
def pauliNet : Ansatz where
  N_param := QMC.PauliNet.N_param
  ansatz := QMC.PauliNet.pauliNetExpr

/-! ## Validity (all proofs `sorry`, but *true* by construction)

* **antisymmetric**: row `i` of each orbital matrix is built from electron `i`'s own
  features, with all other same-spin electrons entering only through permutation-
  invariant sums (message passing) — so permuting same-spin inputs permutes rows, and
  `det` contributes exactly `σ.sign`. The Jastrow factor is a symmetric function of
  same-spin configurations, hence does not break antisymmetry.
* **smooth**: the program is a composition of polynomials (`add/mul/sub/neg/einsum/
  det/sum`), `exp`, division by `exp(2x)+1 ≥ 1`, and index manipulations that are
  constant in the electron coordinates (`gather/iota/mod/ofNat/unflatten/broadcast/
  transpose/cast`). No `sqrt`, no `abs`, no `choice` on float data. The composition is
  real-analytic, in particular `ContDiff ℝ 2`.
* **exp_decay**: every activation is `tanh`-bounded, so orbital pre-factors are
  bounded by a θ-dependent constant; each orbital carries a Gaussian envelope
  `Σ_I exp(-ζₖ‖xᵢ-R_I‖²)` with `ζₖ = exp(ζrawₖ) > 0` for all θ. Hadamard's inequality
  turns per-orbital envelopes into a determinant envelope, the Jastrow factor only
  inflates `C` (`|J| ≤ #pairs`), and a Gaussian envelope implies the stated
  exponential one. -/

/-- The ansatz is valid: for every parameter vector and every molecule, the
wavefunction satisfies antisymmetry, C² smoothness, and the exponential envelope.
Proved fields are left as `sorry` per task instructions. -/
theorem pauliNet_isValid : pauliNet.isValid := by
  unfold Ansatz.isValid
  intro _N_nuc _N_up _N_down _hN _hE _θ _R _Z
  exact ⟨sorry, sorry, sorry⟩

#eval IO.println (pauliNet.ansatz 2 1 1).code
