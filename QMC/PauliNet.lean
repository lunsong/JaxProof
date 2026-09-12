import Xla
import QMC.Ansatz

/-!
# `QMC/PauliNet.lean` — a PauliNet-style molecular QMC ansatz in the Soir/XLA DSL

This file implements a SOTA-family neural-network wavefunction ansatz for molecules,
satisfying the validity contract of `QMC/Ansatz.lean` (`IsValidQMCWavefunction`:
antisymmetry, C² smoothness, exponential envelope). All proofs are `sorry`; the
architecture is chosen so that all three contract fields are *true*, i.e. the sorries
are dischargeable in principle (see the per-field notes at the bottom).

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
  | cons a s ih => simp [Function.comp_def, ih]

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

/-! ## Basic building blocks -/

/-- `tanh` from `exp` and `div` (both have real `DirectImpl` semantics; the `tanh`
prim is a zero stub). Denominator `exp(2x)+1 ≥ 1`, so this is real-analytic. -/
def tanh' {s : Shape} (x : Expr XlaOp args [⟨.float, s⟩]) : Expr XlaOp args [⟨.float, s⟩] :=
  let one : Expr XlaOp args [⟨.float, s⟩] := Xla.ofNat 1
  let two : Expr XlaOp args [⟨.float, s⟩] := Xla.ofNat 2
  let e := Xla.exp (Xla.mul two x)
  Xla.sub one (Xla.div two (Xla.add e one))

/-- Squared pairwise distances `d²[i,j] = ‖xᵢ - yⱼ‖²` (never `sqrt`: C²-safe). -/
def pairDist2 {N M : ℕ} (x : Expr XlaOp args [⟨.float, [N, 3]⟩])
    (y : Expr XlaOp args [⟨.float, [M, 3]⟩]) : Expr XlaOp args [⟨.float, [N, M]⟩] :=
  let xb := bcastInsert [N] [M] [3] x
  let yb := bcastPrepend [N] [M, 3] y
  let d := Xla.sub xb yb
  Xla.sum 1 (Xla.transpose (Xla.mul d d) perm[2,0,1])

/-- Linear layer `y = x Wᵀ + b` on a `[N, d]` particle-feature tensor. -/
def linear {N d d' : ℕ} (W : Expr XlaOp args [⟨.float, [d', d]⟩])
    (b : Expr XlaOp args [⟨.float, [d']⟩])
    (x : Expr XlaOp args [⟨.float, [N, d]⟩]) : Expr XlaOp args [⟨.float, [N, d']⟩] :=
  let y := Xla.einsum [d, N, d'] [[#1, #0], [#2, #0]] 1 (x.append W)
  Xla.add y (Xla.broadcast [⟨N, false⟩, ⟨d', true⟩] b)

/-- Linear layer on the last axis of a `[A, B, K]` pair-feature tensor. -/
def linear3 {A B K d : ℕ} (W : Expr XlaOp args [⟨.float, [d, K]⟩])
    (b : Expr XlaOp args [⟨.float, [d]⟩])
    (x : Expr XlaOp args [⟨.float, [A, B, K]⟩]) : Expr XlaOp args [⟨.float, [A, B, d]⟩] :=
  let y := Xla.einsum [K, A, B, d] [[#1, #2, #0], [#3, #0]] 1 (x.append W)
  Xla.add y (Xla.broadcast [⟨A, false⟩, ⟨B, false⟩, ⟨d, true⟩] b)

/-- Gaussian radial basis of squared distances: `exp(-exp(γₖ)·(d²-μₖ)²)`; appends a
basis axis. Centers/widths are learned (`∀θ`-safe: `exp(γₖ) > 0` always). Bounded and
real-analytic in the electron coordinates. -/
def rbf {s : Shape} {K : ℕ} (μ γ : Expr XlaOp args [⟨.float, [K]⟩])
    (d2 : Expr XlaOp args [⟨.float, s⟩]) : Expr XlaOp args [⟨.float, s ++ [K]⟩] :=
  let d2b := bcastAppend s [K] d2
  let μb := bcastPrepend s [K] μ
  let γb := bcastPrepend s [K] γ
  let r := Xla.sub d2b μb
  Xla.exp (Xla.neg (Xla.mul (Xla.exp γb) (Xla.mul r r)))

/-- SchNet message: `mᵢ = Σⱼ F[i,j,:] ⊙ h[j,:]` as one einsum. -/
def message {N M d : ℕ} (F : Expr XlaOp args [⟨.float, [N, M, d]⟩])
    (h : Expr XlaOp args [⟨.float, [M, d]⟩]) : Expr XlaOp args [⟨.float, [N, d]⟩] :=
  Xla.einsum [M, N, d] [[#1, #0, #2], [#0, #2]] 1 (F.append h)

/-- Row index `k ↦ k mod P` (integer index arithmetic — allowed op discipline). -/
def modIdx (N P : ℕ) : Expr XlaOp args [⟨.int, [N]⟩] :=
  Xla.mod (Xla.iota N) (Xla.ofNat P)

/-- Gather rows of `U : [P, N]` picked by `rowidx : [N]` into an `[N, N]` matrix. -/
def gatherRows {P N : ℕ} (U : Expr XlaOp args [⟨.float, [P, N]⟩])
    (rowidx : Expr XlaOp args [⟨.int, [N]⟩]) : Expr XlaOp args [⟨.float, [N, N]⟩] :=
  let i₀ := bcastAppend [N] [N] rowidx
  let i₁ := bcastPrepend [N] [N] (Xla.iota N)
  Xla.gather U (i₀.append i₁)

/-- Nuclear embeddings: rows of a `Zmax × d` table gathered by atomic number. -/
def embedNuc {Zmax d N_nuc : ℕ} (table : Expr XlaOp args [⟨.float, [Zmax, d]⟩])
    (Z : Expr XlaOp args [⟨.int, [N_nuc]⟩]) : Expr XlaOp args [⟨.float, [N_nuc, d]⟩] :=
  let i₀ := bcastAppend [N_nuc] [d] Z
  let i₁ := bcastPrepend [N_nuc] [d] (Xla.iota d)
  Xla.gather table (i₀.append i₁)

/-- Orbital envelope matrix: `Env[k,i] = Σ_I exp(-ζpool[k]·d²[i,I])`, `ζpool > 0`
(caller passes `exp ζraw`). -/
def envelope {N N_nuc P' : ℕ} (ζpool : Expr XlaOp args [⟨.float, [P']⟩])
    (rowidx : Expr XlaOp args [⟨.int, [N]⟩])
    (d2 : Expr XlaOp args [⟨.float, [N, N_nuc]⟩]) : Expr XlaOp args [⟨.float, [N, N]⟩] :=
  let ζrow : Expr XlaOp args [⟨.float, [N]⟩] := Xla.gather ζpool rowidx
  let ζb := bcastAppend [N] [N, N_nuc] ζrow
  let d2b := bcastPrepend [N] [N, N_nuc] d2
  let e := Xla.exp (Xla.neg (Xla.mul ζb d2b))
  Xla.sum 1 (Xla.transpose e perm[2,0,1])

/-- Slice a shaped parameter tensor out of the flat parameter vector: gather a flat
range `[off, off + shape.prod)` (indices via `iota` arithmetic), then `unflatten`.
Avoids the `dynamic_slice`/`convert_type` zero stubs. -/
def paramAt {P : ℕ} (θ : Expr XlaOp args [⟨.float, [P]⟩]) (shape : Shape) (off : ℕ) :
    Expr XlaOp args [⟨.float, shape⟩] :=
  let off' : Expr XlaOp args [⟨.int, [shape.prod]⟩] := Xla.ofNat off
  let flat : Expr XlaOp args [⟨.float, [shape.prod]⟩] :=
    Xla.gather θ (Xla.add (Xla.iota shape.prod) off')
  Xla.unflatten shape flat

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

/-! ## SchNet interaction block (shared filter/update weights across spin sectors) -/

/-- One message-passing block: for each spin sector, gather electron-electron and
electron-nucleus messages through learned continuous filters of the squared pair
distances, then a tanh-updated linear combination. -/
def schNetBlock {N_up N_down N_nuc : ℕ}
    (θ : Expr XlaOp args [⟨.float, [N_param]⟩]) (off : ℕ)
    (μ γ : Expr XlaOp args [⟨.float, [K]⟩])
    (hu : Expr XlaOp args [⟨.float, [N_up, d]⟩])
    (hd : Expr XlaOp args [⟨.float, [N_down, d]⟩])
    (c : Expr XlaOp args [⟨.float, [N_nuc, d]⟩])
    (d2uu : Expr XlaOp args [⟨.float, [N_up, N_up]⟩])
    (d2ud : Expr XlaOp args [⟨.float, [N_up, N_down]⟩])
    (d2dd : Expr XlaOp args [⟨.float, [N_down, N_down]⟩])
    (d2du : Expr XlaOp args [⟨.float, [N_down, N_up]⟩])
    (d2uN : Expr XlaOp args [⟨.float, [N_up, N_nuc]⟩])
    (d2dN : Expr XlaOp args [⟨.float, [N_down, N_nuc]⟩]) :
    Expr XlaOp args [⟨.float, [N_up, d]⟩] × Expr XlaOp args [⟨.float, [N_down, d]⟩] :=
  -- two-layer continuous filter MLP on the radial basis of a pair-distance tensor
  let filter {A B : ℕ} (oW1 ob1 oW2 ob2 : ℕ) (d2 : Expr XlaOp args [⟨.float, [A, B]⟩]) :
      Expr XlaOp args [⟨.float, [A, B, d]⟩] :=
    let W1 := paramAt θ [d, K] (off + oW1)
    let b1 := paramAt θ [d] (off + ob1)
    let W2 := paramAt θ [d, d] (off + oW2)
    let b2 := paramAt θ [d] (off + ob2)
    tanh' (linear3 W2 b2 (tanh' (linear3 W1 b1 (rbf μ γ d2))))
  let Fuu := filter bo_eeW1 bo_eeb1 bo_eeW2 bo_eeb2 d2uu
  let Fud := filter bo_eeW1 bo_eeb1 bo_eeW2 bo_eeb2 d2ud
  let Fdd := filter bo_eeW1 bo_eeb1 bo_eeW2 bo_eeb2 d2dd
  let Fdu := filter bo_eeW1 bo_eeb1 bo_eeW2 bo_eeb2 d2du
  let Gu := filter bo_enW1 bo_enb1 bo_enW2 bo_enb2 d2uN
  let Gd := filter bo_enW1 bo_enb1 bo_enW2 bo_enb2 d2dN
  let V := paramAt θ [d, d] (off + bo_V)
  let g := paramAt θ [d] (off + bo_g)
  let mu := Xla.add (message Fuu hu) (Xla.add (message Fud hd) (message Gu c))
  let md := Xla.add (message Fdd hd) (Xla.add (message Fdu hu) (message Gd c))
  (tanh' (Xla.add (linear V g hu) mu), tanh' (Xla.add (linear V g hd) md))

/-! ## Determinant head (shared across spin sectors, CRT-pooled orbitals) -/

/-- One determinant of pooled orbitals for a spin sector:
`Φ[k,i] = (A·tanh(V hᵢ + c))_{k mod P} · Σ_I exp(-exp(ζraw_{k mod Q})‖xᵢ-R_I‖²)`.
Row `k` is a distinct function of `xᵢ` for `k < P·Q` (CRT), with fixed parameter
count. -/
def detHead {N N_nuc : ℕ}
    (θ : Expr XlaOp args [⟨.float, [N_param]⟩]) (off : ℕ)
    (h : Expr XlaOp args [⟨.float, [N, d]⟩])
    (d2nuc : Expr XlaOp args [⟨.float, [N, N_nuc]⟩]) :
    Expr XlaOp args [⟨.float, []⟩] :=
  let Vm := paramAt θ [d, d] (off + ho_Vm)
  let cm := paramAt θ [d] (off + ho_cm)
  let Am := paramAt θ [P, d] (off + ho_Am)
  let ζraw := paramAt θ [Q] (off + ho_ζ)
  let g := tanh' (linear Vm cm h)
  let U := Xla.einsum [d, P, N] [[#2, #0], [#1, #0]] 1 (g.append Am)
  let M := gatherRows U (modIdx N P)
  let Env := envelope (Xla.exp ζraw) (modIdx N Q) d2nuc
  Xla.det (Xla.mul M Env)

/-- Bounded Jastrow over one pair set: `Σ_{i,j} tanh(wJ · e(d²ᵢⱼ))`. -/
def jastrow {A B : ℕ} (wJ μ γ : Expr XlaOp args [⟨.float, [K]⟩])
    (d2 : Expr XlaOp args [⟨.float, [A, B]⟩]) : Expr XlaOp args [⟨.float, []⟩] :=
  let e := rbf μ γ d2
  let v := Xla.einsum [K, A, B] [[#1, #2, #0], [#0]] 1 (e.append wJ)
  Xla.sum 2 (tanh' v)

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
    let h0 := paramAt θ [d] off_h0
    let table := paramAt θ [Zmax, d] off_table
    let μ := paramAt θ [K] off_μ
    let γ := paramAt θ [K] off_γ
    -- nuclear embeddings and squared pair distances
    let c := embedNuc table Z
    let d2uu := pairDist2 xu xu
    let d2ud := pairDist2 xu xd
    let d2dd := pairDist2 xd xd
    let d2du := pairDist2 xd xu
    let d2uN := pairDist2 xu R
    let d2dN := pairDist2 xd R
    -- SchNet embedding: L = 2 blocks, constant initial electron features
    let hu0 := bcastPrepend [N_up] [d] h0
    let hd0 := bcastPrepend [N_down] [d] h0
    let (hu1, hd1) := schNetBlock θ (off_block 0) μ γ hu0 hd0 c d2uu d2ud d2dd d2du d2uN d2dN
    let (hu2, hd2) := schNetBlock θ (off_block 1) μ γ hu1 hd1 c d2uu d2ud d2dd d2du d2uN d2dN
    -- D = 2 determinants per spin sector, heads shared across sectors
    let du0 := detHead θ (off_head 0) hu2 d2uN
    let dd0 := detHead θ (off_head 0) hd2 d2dN
    let du1 := detHead θ (off_head 1) hu2 d2uN
    let dd1 := detHead θ (off_head 1) hd2 d2dN
    let c0 := paramAt θ [] off_c
    let c1 := paramAt θ [] (off_c + 1)
    -- bounded Jastrow over all electron pairs
    let wJ := paramAt θ [K] off_wJ
    let J := Xla.add (jastrow wJ μ γ d2uu)
      (Xla.add (jastrow wJ μ γ d2ud) (jastrow wJ μ γ d2dd))
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
