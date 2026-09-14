import Random

/-!
# Sampling with `Random`

A `Random.SimpleExpr` is a program that draws independent random numbers and
combines them arithmetically.  Its semantics is a `RandVar`, a real random
variable, and `RandVar.mean` computes its expectation.

Draws are addressed by *keys*: a key is an `ℕ` passed as an expression
argument, and `shuffle k = k + 1` derives a fresh key.  Thus `normal k` and
`normal (shuffle k)` are independent.  Since keys are explicit arguments,
every expression records the keys it consumes — `z0` below has type
`SimpleExpr [.key] .data` and is evaluated at a key `k : ℕ`.

The examples evaluate programs with the `reduce_random` and `reduce_soir`
simp sets and then integrate with the `mean` lemmas.
-/

open Random Soir
open MeasureTheory (Integrable)

/-! ## Key-indexed draws -/

/-- A standard normal draw at the expression's key. -/
def z0 : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k => Random.normal k

/-- A standard normal draw at `shuffle k`, independent of `z0 k`. -/
def z1 : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k => Random.normal (Random.shuffle k)

/-- A uniform draw on `[0, 1]` at the expression's key. -/
def u0 : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k => Random.uniform k

/-- A uniform draw on `[0, 1]` at `shuffle k`, independent of `u0 k`. -/
def u1 : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k => Random.uniform (Random.shuffle k)

/-! ## Means of the primitive draws -/

/-- The standard normal has mean zero. -/
theorem mean_z0 (k : ℕ) : (z0.eval k).mean = 0 := by
  simp [z0, reduce_random, reduce_soir]

/-- The uniform law on `[0, 1]` has mean `1/2`. -/
theorem mean_u0 (k : ℕ) : (u0.eval k).mean = 1 / 2 := by
  simp [u0, reduce_random, reduce_soir]

/-! ## Linearity: Monte-Carlo averages are unbiased -/

/-- The two-sample average of independent standard normals. -/
def zbar : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k =>
    Random.div (Random.add (z0.apply k) (z1.apply k)) (Random.ofNat 2)

/-- Its mean is zero. -/
theorem mean_zbar (k : ℕ) : (zbar.eval k).mean = 0 := by
  simp [zbar, z0, z1, reduce_random, reduce_soir]

/-- The two-sample average of independent uniforms. -/
def ubar : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k =>
    Random.div (Random.add (u0.apply k) (u1.apply k)) (Random.ofNat 2)

/-- Its mean is `1/2`, the integral `∫₀¹ x dx`. -/
theorem mean_ubar (k : ℕ) : (ubar.eval k).mean = 1 / 2 := by
  simp [ubar, u0, u1, reduce_random, reduce_soir]

/-! ## The mean does not see nonlinearities -/

/-- `E[Z²] = 1`, although `(E[Z])² = 0`. -/
def zsq : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k => Random.mul (Random.normal k) (Random.normal k)

theorem mean_zsq (k : ℕ) : (zsq.eval k).mean = 1 := by
  simp [zsq, reduce_random, reduce_soir]

/-- `E[U²] = 1/3`, the exact value of `∫₀¹ x² dx`. -/
def usq : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k => Random.mul (Random.uniform k) (Random.uniform k)

theorem mean_usq (k : ℕ) : (usq.eval k).mean = 1 / 3 := by
  simp [usq, reduce_random, reduce_soir]

/-! ## Independence -/

/-- Independent draws factor through the mean: `E[Zₖ Zₗ] = E[Zₖ] E[Zₗ] = 0`. -/
def zprod : Random.SimpleExpr [.key, .key] .data :=
  Soir.Expr.ofFn fun k l => Random.mul (Random.normal k) (Random.normal l)

theorem mean_zprod (k l : ℕ) (h : k ≠ l) : (zprod.eval k l).mean = 0 := by
  simpa [zprod, reduce_random, reduce_soir] using Random.RandVar.mean_mul_normal_of_ne h

/-- The sample average of two squared normals, an unbiased estimator of `E[Z²]`. -/
def zsqbar : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k =>
    Random.div
      (Random.add (Random.mul (z0.apply k) (z0.apply k))
        (Random.mul (z1.apply k) (z1.apply k)))
      (Random.ofNat 2)

theorem mean_zsqbar (k : ℕ) : (zsqbar.eval k).mean = 1 := by
  simp [zsqbar, z0, z1, reduce_random, reduce_soir]

/-! ## A martingale step

Adding independent, zero-mean noise does not change an expectation.  This is
why an Euler-Maruyama step `X + μ dt + σ √dt Z` is unbiased for its drift. -/

/-- `(k, x) ↦ x + 5 Z` for a standard normal `Z` at key `k`. -/
def noiseStep : Random.SimpleExpr [.key, .data] .data :=
  Soir.Expr.ofFn fun k x =>
    Random.add x (Random.mul (Random.ofNat 5) (Random.normal k))

theorem mean_noiseStep (k : ℕ) (x : Random.RandVar) (hx : Integrable x Random.law) :
    (noiseStep.eval k x).mean = x.mean := by
  simp [noiseStep, reduce_random, reduce_soir, hx]

def noiseMul (f : Random.SimpleExpr [.key] .data) : Random.SimpleExpr [.key] .data :=
  Soir.Expr.ofFn fun k =>
    let x₀ := Random.normal k;
    let k := Random.shuffle k;
    let x₁ := f.apply k;
    Random.mul x₀ x₁

theorem mean_noiseMul (k : ℕ) (f : Random.SimpleExpr [.key] .data) :
    ((noiseMul f).eval k).mean = 0 := sorry

