import QMC.FermiNet

/-!
# FermiNet — apply style

The same program as `QMC/FermiNet.lean`, with the reusable components written the
other way round. There each component is a *builder-style* helper: a function
`Expr → Expr` that inlines its body at every call site (`Examples/CoulombBuilder.lean`
is the minimal example). Here each component is a closed `SimpleExpr` whose inputs are
formal arguments, and the components are composed with `Soir.Expr.apply`
(`Examples/Coulomb.lean` is the minimal example) — so the emitted IR is a set of
`@n` library bodies called by `call; @n, …` instead of one flat program.

Why this matters for code generation cost: `Expr.code` (Soir/Core.lean) deduplicates
sub-expressions by `Expr.hashExpr`, a *structural* hash that re-walks a node's whole
subtree (calling `toString` on the op of every `bind` node), and it evaluates that
hash many times per node — once per cache element in the `List.find?` lookups of
`Expr.genCode`/`Expr.addLib`, and again in `Expr.addVars`. Cost is therefore
(entries in the scope's cache) × (hash of a subtree), so the size of the scope a
program is generated in matters quadratically. A builder-style program is one flat
scope of every binding it contains; an apply-style program is a short chain of calls
whose library bodies are each generated in a scope of their own small size.

Measured (Lean 4.33.1, `lake env lean`, interpreter: DAG construction + `Expr.code`,
so the numbers below exclude the ~2.5 s toolchain startup):

| program | `QMC/FermiNet.lean` (builder) | this file (apply) |
|---|---|---|
| `fermiNetAnsatz 2 1 1` (H₂)  | 17.0 s | 0.37 s |
| `fermiNetAnsatz 1 2 1` (Li)  | 17.2 s | 0.40 s |
| `fermiNetAnsatz 1 1 2` (Be)  | 17.4 s | 0.40 s |
| `fermiNetAnsatz 3 4 4` (H₂O) | 16.9 s | 0.37 s |

Both are flat in molecule size (the program structure, not the tensor shapes, drives
the cost). The emitted IR for H₂O is one 360-binding flat program for the builder
style versus a 31-binding top level calling 59 libraries here — the scope that the
`find?` scans run over shrinks from 360 bindings to 31, and each library body is
generated in its own (small) scope. For scale: one full structural hash of this
program takes 3 ms, so the builder-style codegen does ~5 700 full-program hashes'
worth of work against ~120 for the apply style.

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

The two IRs evaluate to the *same* amplitude in the JAX evaluator (bit-identical for
the four molecules above: `python3 python/test_ferminet.py` and
`FERMINET_EMITTER=emit_ferminet_apply_ir.lean python3 python/test_ferminet.py`).

The library bodies below call the same primitive builders as `QMC/FermiNet.lean`
(imported from it), so the two programs compute the same function; only the emitted
IR differs. `python/emit_ferminet_apply_ir.lean` emits this program for the Python
harness (`FERMINET_EMITTER=emit_ferminet_apply_ir.lean python3 python/test_ferminet.py`).
-/

namespace FermiNetApply

open Soir Xla FermiNet

/-! ### Libraries for the primitive compositions -/

/-- `tanh x = 1 - 2 / (exp (2x) + 1)`. -/
def tanhL (s : Shape) : SimpleExpr [⟨.float, s⟩] ⟨.float, s⟩ :=
  Expr.ofFn fun x => tanh x

/-- Slice `len` consecutive floats of the parameter vector from `off`. -/
def paramSliceL (off len : ℕ) : SimpleExpr [⟨.float, [N_PARAM]⟩] ⟨.float, [len]⟩ :=
  Expr.ofFn fun θ => paramSlice off len θ

/-- Reshape a parameter slice into shape `s`. -/
def paramBlockL (off : ℕ) (s : Shape) : SimpleExpr [⟨.float, [N_PARAM]⟩] ⟨.float, s⟩ :=
  Expr.ofFn fun θ => Xla.unflatten s ((paramSliceL off s.prod).apply θ)

/-- A scalar parameter reparameterized through `exp`, so it is `> 0` for every `θ`. -/
def posScalarL (off : ℕ) : SimpleExpr [⟨.float, [N_PARAM]⟩] ⟨.float, []⟩ :=
  Expr.ofFn fun θ => Xla.exp (Xla.sum 1 ((paramSliceL off 1).apply θ))

/-! ### Geometry -/

/-- Electron–nucleus displacement `rᵢ - Rα`, shape `[N, N_nuc, 3]`. -/
def enDispL (N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_nuc, 3]⟩] ⟨.float, [N, N_nuc, 3]⟩ :=
  Expr.ofFn fun r => fun R => enDisp N N_nuc r R

/-- Electron–nucleus distance `√(‖rᵢ - Rα‖² + ε)`. -/
def enDistL (N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_nuc, 3]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N_nuc]⟩ :=
  Expr.ofFn fun r => fun R => fun θ =>
    let d := (enDispL N N_nuc).apply (r.append R)
    let d2 := Xla.mul d d
    let r2 := Xla.sum 1 (Xla.transpose d2 perm[2, 0, 1])
    let eps := (posScalarL OFF_EPS).apply θ
    Xla.sqrt (Xla.add r2 (Xla.broadcast [⟨N, false⟩, ⟨N_nuc, false⟩] eps))

/-- Same-spin electron–electron displacement `rⱼ - rᵢ`, shape `[N, N, 3]`. -/
def pairDispL (N : ℕ) : SimpleExpr [⟨.float, [N, 3]⟩] ⟨.float, [N, N, 3]⟩ :=
  Expr.ofFn fun r => pairDisp N r

/-- Same-spin electron–electron distance `√(‖rⱼ - rᵢ‖² + ε)`. -/
def pairDistL (N : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, [N, N]⟩ :=
  Expr.ofFn fun r => fun θ =>
    let d := (pairDispL N).apply r
    let d2 := Xla.mul d d
    let r2 := Xla.sum 1 (Xla.transpose d2 perm[2, 0, 1])
    let eps := (posScalarL OFF_EPS).apply θ
    Xla.sqrt (Xla.add r2 (Xla.broadcast [⟨N, false⟩, ⟨N, false⟩] eps))

/-- Opposite-spin displacement `r'ⱼ - rᵢ`, shape `[N, N', 3]`. -/
def pairDispCrossL (N N' : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N', 3]⟩] ⟨.float, [N, N', 3]⟩ :=
  Expr.ofFn fun r => fun r' => pairDispCross N N' r r'

/-- Opposite-spin distance `√(‖r'ⱼ - rᵢ‖² + ε)`. -/
def pairDistCrossL (N N' : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N', 3]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N']⟩ :=
  Expr.ofFn fun r => fun r' => fun θ =>
    let d := (pairDispCrossL N N').apply (r.append r')
    let d2 := Xla.mul d d
    let r2 := Xla.sum 1 (Xla.transpose d2 perm[2, 0, 1])
    let eps := (posScalarL OFF_EPS).apply θ
    Xla.sqrt (Xla.add r2 (Xla.broadcast [⟨N, false⟩, ⟨N', false⟩] eps))

/-! ### The three streams -/

/-- Initial one-electron stream. -/
def oneStreamInitL (N N_nuc : ℕ) :
    SimpleExpr
      [⟨.float, [N, 3]⟩, ⟨.float, [N_nuc, 3]⟩, ⟨.int, [N_nuc]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, F]⟩ :=
  Expr.ofFn fun r => fun R => fun Z => fun θ =>
    let disp := (enDispL N N_nuc).apply (r.append R)
    let dist := (enDistL N N_nuc).apply ((r.append R).append θ)
    let ztab := (paramBlockL OFF_ZTAB [Z_TAB]).apply θ
    let Zc : Expr XlaOp _ [⟨.int, [N_nuc]⟩] := Xla.mod Z (Xla.ofNat Z_TAB)
    let zRaw := Xla.gather (α := .float) (s := [Z_TAB]) (s' := [N_nuc]) ztab Zc
    let wd := (paramBlockL OFF_WD [N_nuc, 3, F]).apply θ
    let ws := (paramBlockL OFF_WS [N_nuc, F]).apply θ
    let wz := (paramBlockL OFF_WZ [N_nuc, F]).apply θ
    let dispT := Xla.transpose disp perm[1, 2, 0]
    let distT := Xla.transpose dist perm[1, 0]
    let zB := Xla.broadcast [⟨N, false⟩, ⟨N_nuc, true⟩] zRaw
    let zT := Xla.transpose zB perm[1, 0]
    let tD := Xla.einsum (s := [N_nuc, 3, N, F]) [[#0, #1, #2], [#0, #1, #3]] 2 (dispT.append wd)
    let tS := Xla.einsum (s := [N_nuc, N, F]) [[#0, #1], [#0, #2]] 1 (distT.append ws)
    let tZ := Xla.einsum (s := [N_nuc, N, F]) [[#0, #1], [#0, #2]] 1 (zT.append wz)
    let b0 := Xla.broadcast [⟨N, false⟩, ⟨F, true⟩] ((paramSliceL OFF_B0 F).apply θ)
    (tanhL [N, F]).apply (Xla.add (Xla.add (Xla.add tD tS) tZ) b0)

/-- Initial same-spin two-electron stream. -/
def twoStreamInitL (N : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, [N, N, F]⟩ :=
  Expr.ofFn fun r => fun θ =>
    let pd := (pairDispL N).apply r
    let pd2 := (pairDistL N).apply (r.append θ)
    let we := (paramBlockL OFF_WE [3, F]).apply θ
    let wd := (paramSliceL OFF_WD2 F).apply θ
    let pdT := Xla.transpose pd perm[2, 0, 1]
    let tD := Xla.einsum (s := [3, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (pdT.append we)
    let wdB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] wd
    let pd2B := Xla.broadcast [⟨N, true⟩, ⟨N, true⟩, ⟨F, false⟩] pd2
    let tS := Xla.mul pd2B wdB
    let gb := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] ((paramSliceL OFF_GB F).apply θ)
    (tanhL [N, N, F]).apply (Xla.add (Xla.add tD tS) gb)

/-- Initial opposite-spin two-electron stream. -/
def twoStreamInitCrossL (N N' : ℕ) :
    SimpleExpr [⟨.float, [N, 3]⟩, ⟨.float, [N', 3]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N', F]⟩ :=
  Expr.ofFn fun r => fun r' => fun θ =>
    let pd := (pairDispCrossL N N').apply (r.append r')
    let pd2 := (pairDistCrossL N N').apply ((r.append r').append θ)
    let we := (paramBlockL OFF_WE [3, F]).apply θ
    let wd := (paramSliceL OFF_WD2 F).apply θ
    let pdT := Xla.transpose pd perm[2, 0, 1]
    let tD := Xla.einsum (s := [3, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (pdT.append we)
    let wdB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] wd
    let pd2B := Xla.broadcast [⟨N, true⟩, ⟨N', true⟩, ⟨F, false⟩] pd2
    let tS := Xla.mul pd2B wdB
    let gb := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] ((paramSliceL OFF_GB F).apply θ)
    (tanhL [N, N', F]).apply (Xla.add (Xla.add tD tS) gb)

/-! ### Layer updates -/

/-- One-electron stream update: `h ← tanh(V h + Σⱼ w ⊙ g_ij + Σⱼ w' ⊙ g_ij^{σσ̄} + b)`. -/
def oneStreamLayerL (N N' off : ℕ) :
    SimpleExpr
      [⟨.float, [N, F]⟩, ⟨.float, [N, N, F]⟩, ⟨.float, [N, N', F]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, F]⟩ :=
  Expr.ofFn fun h => fun g => fun gC => fun θ =>
    let V := (paramBlockL off [F, F]).apply θ
    let w := (paramSliceL (off + 1024) F).apply θ
    let wC := (paramSliceL (off + 1056) F).apply θ
    let b := (paramSliceL (off + 1088) F).apply θ
    let tV := Xla.einsum (s := [F, N, F]) [[#1, #0], [#0, #2]] 1 (h.append V)
    let tG := Xla.einsum (s := [N, N, F]) [[#0, #1, #2], [#2]] 1 (g.append w)
    let gCT := Xla.transpose gC perm[1, 0, 2]
    let tGC := Xla.einsum (s := [N', N, F]) [[#0, #1, #2], [#2]] 1 (gCT.append wC)
    let bB := Xla.broadcast [⟨N, false⟩, ⟨F, true⟩] b
    (tanhL [N, F]).apply (Xla.add (Xla.add (Xla.add tV tG) tGC) bB)

/-- Same-spin two-electron stream update: `g ← tanh(G g + H (h_i + h_j) + c)`. -/
def twoStreamLayerL (N off : ℕ) :
    SimpleExpr [⟨.float, [N, N, F]⟩, ⟨.float, [N, F]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N, F]⟩ :=
  Expr.ofFn fun g => fun h => fun θ =>
    let G := (paramBlockL (off + 1120) [F, F]).apply θ
    let H := (paramBlockL (off + 2144) [F, F]).apply θ
    let c := (paramSliceL (off + 3168) F).apply θ
    let gT := Xla.transpose g perm[2, 0, 1]
    let tG := Xla.einsum (s := [F, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (gT.append G)
    let hA := Xla.broadcast [⟨N, false⟩, ⟨N, true⟩, ⟨F, true⟩] h
    let hB := Xla.broadcast [⟨N, true⟩, ⟨N, false⟩, ⟨F, true⟩] h
    let hS := Xla.transpose (Xla.add hA hB) perm[2, 0, 1]
    let tH := Xla.einsum (s := [F, N, N, F]) [[#0, #1, #2], [#0, #3]] 1 (hS.append H)
    let cB := Xla.broadcast [⟨N, false⟩, ⟨N, false⟩, ⟨F, true⟩] c
    (tanhL [N, N, F]).apply (Xla.add (Xla.add tG tH) cB)

/-- Opposite-spin two-electron stream update:
`g^{σσ̄} ← tanh(G' g^{σσ̄} + H' (h_i^σ + h_j^{σ̄}) + c')`. -/
def twoStreamLayerCrossL (N N' off : ℕ) :
    SimpleExpr [⟨.float, [N, N', F]⟩, ⟨.float, [N, F]⟩, ⟨.float, [N', F]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [N, N', F]⟩ :=
  Expr.ofFn fun gC => fun h => fun h' => fun θ =>
    let G := (paramBlockL (off + 3200) [F, F]).apply θ
    let H := (paramBlockL (off + 4224) [F, F]).apply θ
    let c := (paramSliceL (off + 5248) F).apply θ
    let gT := Xla.transpose gC perm[2, 0, 1]
    let tG := Xla.einsum (s := [F, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (gT.append G)
    let hA := Xla.broadcast [⟨N, true⟩, ⟨N', false⟩, ⟨F, true⟩] h
    let hB := Xla.broadcast [⟨N, false⟩, ⟨N', true⟩, ⟨F, true⟩] h'
    let hS := Xla.transpose (Xla.add hA hB) perm[2, 0, 1]
    let tH := Xla.einsum (s := [F, N, N', F]) [[#0, #1, #2], [#0, #3]] 1 (hS.append H)
    let cB := Xla.broadcast [⟨N, false⟩, ⟨N', false⟩, ⟨F, true⟩] c
    (tanhL [N, N', F]).apply (Xla.add (Xla.add tG tH) cB)

/-! ### Outputs -/

/-- Bounded Jastrow factor `J = Σᵢ w_J · hᵢ` (scalar). -/
def jastrowL (N : ℕ) :
    SimpleExpr [⟨.float, [N, F]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, []⟩ :=
  Expr.ofFn fun h => fun θ =>
    let wJ := (paramSliceL OFF_WJ F).apply θ
    Xla.einsum (s := [N, F]) [[#0, #1], [#1]] 2 (h.append wJ)

/-- Per-orbital exponential envelope, shape `[K, N, N]`. -/
def envelopeL (N N_nuc : ℕ) :
    SimpleExpr [⟨.float, [N, N_nuc]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, [K, N, N]⟩ :=
  Expr.ofFn fun dist => fun θ =>
    let A := (paramBlockL OFF_A [K, N, N_nuc]).apply θ
    let Aexp := Xla.exp A
    let dT := Xla.transpose dist perm[1, 0]
    let e := Xla.einsum (s := [N_nuc, K, N, N]) [[#1, #2, #0], [#0, #3]] 1 (Aexp.append dT)
    Xla.transpose e perm[0, 2, 1]

/-- Orbital matrix of shape `[K, N, N]`. -/
def orbitalMatrixL (N : ℕ) :
    SimpleExpr [⟨.float, [N, F]⟩, ⟨.float, [K, N, N]⟩, ⟨.float, [N_PARAM]⟩]
      ⟨.float, [K, N, N]⟩ :=
  Expr.ofFn fun h => fun envExp => fun θ =>
    let worb := (paramBlockL OFF_WORB [F, K, N]).apply θ
    let φ := Xla.einsum (s := [F, N, K, N]) [[#1, #0], [#0, #2, #3]] 1 (h.append worb)
    let φT := Xla.transpose φ perm[1, 0, 2]
    Xla.mul φT envExp

/-- Weighted sum of the `K` determinants of a spin sector: `Σₖ exp w_k · detₖ`. -/
def detBlockL (N : ℕ) :
    SimpleExpr [⟨.float, [K, N, N]⟩, ⟨.float, [N_PARAM]⟩] ⟨.float, []⟩ :=
  Expr.ofFn fun orb => fun θ =>
    let dets := Xla.vmap (batch := K) (ins := [⟨.float, [N, N]⟩]) (outs := [⟨.float, []⟩])
      (detExpr N) orb .nil
    let wdet := Xla.exp ((paramSliceL OFF_WDET K).apply θ)
    Xla.sum 1 (Xla.mul dets wdet)

/-! ### The ansatz -/

/-- The apply-style FermiNet program for a molecule with `N_nuc` nuclei and
`N_up` / `N_down` electrons. -/
def fermiNetAnsatzApply (N_nuc N_up N_down : ℕ) : SimpleExpr
    [⟨.float, [N_PARAM]⟩, -- parameters
      ⟨.float, [N_nuc, 3]⟩, -- positions of nuclei
      ⟨.int, [N_nuc]⟩, -- types of nuclei
      ⟨.float, [N_up, 3]⟩, -- positions of spin up electrons
      ⟨.float, [N_down, 3]⟩] -- positions of spin down electrons
    ⟨.float, []⟩ :=
  Soir.Expr.ofFn fun θ => fun R => fun Z => fun rUp => fun rDown =>
    let hU0 := (oneStreamInitL N_up N_nuc).apply (((rUp.append R).append Z).append θ)
    let hD0 := (oneStreamInitL N_down N_nuc).apply (((rDown.append R).append Z).append θ)
    let gU0 := (twoStreamInitL N_up).apply (rUp.append θ)
    let gD0 := (twoStreamInitL N_down).apply (rDown.append θ)
    let gUD0 := (twoStreamInitCrossL N_up N_down).apply ((rUp.append rDown).append θ)
    let gDU0 := (twoStreamInitCrossL N_down N_up).apply ((rDown.append rUp).append θ)
    -- layer 0
    let hU1 := (oneStreamLayerL N_up N_down OFF_L0).apply
      (((hU0.append gU0).append gUD0).append θ)
    let hD1 := (oneStreamLayerL N_down N_up OFF_L0).apply
      (((hD0.append gD0).append gDU0).append θ)
    let gU1 := (twoStreamLayerL N_up OFF_L0).apply ((gU0.append hU0).append θ)
    let gD1 := (twoStreamLayerL N_down OFF_L0).apply ((gD0.append hD0).append θ)
    let gUD1 := (twoStreamLayerCrossL N_up N_down OFF_L0).apply
      (((gUD0.append hU0).append hD0).append θ)
    let gDU1 := (twoStreamLayerCrossL N_down N_up OFF_L0).apply
      (((gDU0.append hD0).append hU0).append θ)
    -- layer 1
    let hU2 := (oneStreamLayerL N_up N_down OFF_L1).apply
      (((hU1.append gU1).append gUD1).append θ)
    let hD2 := (oneStreamLayerL N_down N_up OFF_L1).apply
      (((hD1.append gD1).append gDU1).append θ)
    -- Jastrow factor (the streams are tanh-bounded, so `exp J` is bounded)
    let JU := (jastrowL N_up).apply (hU2.append θ)
    let JD := (jastrowL N_down).apply (hD2.append θ)
    let J := Xla.exp (Xla.add JU JD)
    -- exponential envelopes
    let distU := (enDistL N_up N_nuc).apply ((rUp.append R).append θ)
    let distD := (enDistL N_down N_nuc).apply ((rDown.append R).append θ)
    let envU := Xla.exp (Xla.neg ((envelopeL N_up N_nuc).apply (distU.append θ)))
    let envD := Xla.exp (Xla.neg ((envelopeL N_down N_nuc).apply (distD.append θ)))
    -- determinants
    let orbU := (orbitalMatrixL N_up).apply ((hU2.append envU).append θ)
    let orbD := (orbitalMatrixL N_down).apply ((hD2.append envD).append θ)
    let detU := (detBlockL N_up).apply (orbU.append θ)
    let detD := (detBlockL N_down).apply (orbD.append θ)
    Xla.mul (Xla.mul J detU) detD

/-- The apply-style FermiNet `Ansatz`. -/
def fermiNetApply : Ansatz where
  N_param := N_PARAM
  ansatz := fun N_nuc N_up N_down => fermiNetAnsatzApply N_nuc N_up N_down

/-! ### The validity contract (all proofs `sorry`)

Identical statements to `QMC/FermiNet.lean`; the two programs are the same function,
so the proof sketch in that file's docstring applies verbatim. -/

theorem antisymmetric (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNetApply.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    IsAntisymmetric ((fermiNetApply.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc) := by
  sorry

theorem smooth (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNetApply.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    ContDiff ℝ 2
      (Function.uncurry ((fermiNetApply.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc)) := by
  sorry

theorem exp_decay (N_nuc N_up N_down : ℕ) (_ : N_nuc ≠ 0) (_ : N_up + N_down ≠ 0)
    (θ : Fin fermiNetApply.N_param → ℝ) (R_nuc : Xla.Tensor ℝ [N_nuc, 3])
    (Z_nuc : Xla.Tensor ℤ [N_nuc]) :
    ∃ C k : ℝ, 0 < k ∧
      ∀ x, |Function.uncurry ((fermiNetApply.ansatz N_nuc N_up N_down).eval θ R_nuc Z_nuc) x| ≤
        C * Real.exp (-k * ‖x‖) := by
  sorry

/-- FermiNet (apply style) satisfies the QMC validity contract. -/
theorem isValid : fermiNetApply.isValid := by
  intro N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
  exact {
    antisymmetric := antisymmetric N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
    smooth := smooth N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
    exp_decay := exp_decay N_nuc N_up N_down hnuc he θ R_nuc Z_nuc
  }

end FermiNetApply
