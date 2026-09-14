import Random

/-!
# Sampling with `Random`

A `Random.SimpleExpr` is a program that draws independent random numbers and
combines them arithmetically.  Its semantics is a `RandVar`, a real random
variable, and `RandVar.mean` computes its expectation.

Draws come in two flavours:

* `normal k` is a standard normal draw,
* `uniform k` is uniform on `[0, 1]`,

where the key `k` labels the draw.  `key` is the key `0` and `shuffle k` is
`k + 1`, so `normal key` and `normal (shuffle key)` are independent.

The examples below evaluate programs with the `reduce_random` and `reduce_soir`
simp sets and then integrate with the `mean` lemmas.
-/

open Random Soir
open MeasureTheory (Integrable)

/-- A standard normal draw, key `0`. -/
def z0 : Random.SimpleExpr [] .data := Random.normal Random.key

/-- An independent standard normal draw, key `1`. -/
def z1 : Random.SimpleExpr [] .data := Random.normal (Random.shuffle Random.key)

/-- A uniform draw on `[0, 1]`, key `0`. -/
def u0 : Random.SimpleExpr [] .data := Random.uniform Random.key

/-- An independent uniform draw on `[0, 1]`, key `1`. -/
def u1 : Random.SimpleExpr [] .data := Random.uniform (Random.shuffle Random.key)

/-! ## Means of the primitive draws -/

/-- The standard normal has mean zero. -/
theorem mean_z0 : z0.eval.mean = 0 := by
  simp [z0, reduce_random, reduce_soir]

/-- The uniform law on `[0, 1]` has mean `1/2`. -/
theorem mean_u0 : u0.eval.mean = 1 / 2 := by
  simp [u0, reduce_random, reduce_soir]

/-! ## Linearity: Monte-Carlo averages are unbiased -/

/-- The two-sample average of independent standard normals. -/
def zbar : Random.SimpleExpr [] .data :=
  Random.div (Random.add z0 z1) (Random.ofNat 2)

/-- Its mean is zero. -/
theorem mean_zbar : zbar.eval.mean = 0 := by
  simp [zbar, z0, z1, reduce_random, reduce_soir]

/-- The two-sample average of independent uniforms. -/
def ubar : Random.SimpleExpr [] .data :=
  Random.div (Random.add u0 u1) (Random.ofNat 2)

/-- Its mean is `1/2`, the integral `∫₀¹ x dx`. -/
theorem mean_ubar : ubar.eval.mean = 1 / 2 := by
  simp [ubar, u0, u1, reduce_random, reduce_soir]

/-! ## The mean does not see nonlinearities -/

/-- `E[Z²] = 1`, although `(E[Z])² = 0`. -/
def zsq : Random.SimpleExpr [] .data := Random.mul z0 z0

theorem mean_zsq : zsq.eval.mean = 1 := by
  simp [zsq, z0, reduce_random, reduce_soir]

/-- `E[U²] = 1/3`, the exact value of `∫₀¹ x² dx`. -/
def usq : Random.SimpleExpr [] .data := Random.mul u0 u0

theorem mean_usq : usq.eval.mean = 1 / 3 := by
  simp [usq, u0, reduce_random, reduce_soir]

/-! ## Independence -/

/-- Independent draws factor through the mean: `E[Z₀ Z₁] = E[Z₀] E[Z₁] = 0`. -/
def zprod : Random.SimpleExpr [] .data := Random.mul z0 z1

theorem mean_zprod : zprod.eval.mean = 0 := by
  simp (config := { decide := true }) [zprod, z0, z1, reduce_random, reduce_soir]

/-- The sample average of two squared normals, an unbiased estimator of `E[Z²]`. -/
def zsqbar : Random.SimpleExpr [] .data :=
  Random.div (Random.add (Random.mul z0 z0) (Random.mul z1 z1)) (Random.ofNat 2)

theorem mean_zsqbar : zsqbar.eval.mean = 1 := by
  simp [zsqbar, z0, z1, reduce_random, reduce_soir]

/-! ## A martingale step

Adding independent, zero-mean noise does not change an expectation.  This is
why an Euler-Maruyama step `X + μ dt + σ √dt Z` is unbiased for its drift. -/

/-- `X ↦ X + 5 Z` for a fresh standard normal `Z`. -/
def noiseStep : Random.SimpleExpr [.data] .data :=
  Soir.Expr.ofFn fun x =>
    Random.add x (Random.mul (Random.ofNat 5) (Random.normal Random.key))

theorem mean_noiseStep (x : Random.RandVar) (hx : Integrable x Random.law) :
    (noiseStep.eval x).mean = x.mean := by
  simp [noiseStep, reduce_random, reduce_soir, hx]
