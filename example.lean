import JaxProof.Api
import Mathlib.Data.Matrix.Mul

open Jax.Impl

jax_def bodyfn(i, a):
  return a * a

-- define the jax function
jax_def f(n, x):
  return fori_loop n x bodyfn

-- extract python code
#eval IO.println ((Jax.trace f).outward.code)

/-
The output is
def x0(_x0, x1):
  x2 = x1 * x1
  return x2
def x1(x2, x3):
  x4 = jax.lax.fori_loop(0, x2[0], (lambda i, c: x0(jax.numpy.array([i]), c)), x3)
  return x4
-/

-- f(n, x) = x ^ (2 ^ n)
-- prove mathematical properties
example (n : ℕ) (x : List ℝ) :
    (Jax.native f) (.int [n]) (.float x) = .float (x.map (· ^ (2 ^ n))) := by
  simp[f, bodyfn]
  induction n with
  | zero => simp
  | succ n ih =>
    simp [ih]
    simp [HMul.hMul, Jax.Array.pairwise, Jax.Array.mul]
    congr
    apply List.ext_get
    · simp
    simp
    intro i _ _
    conv_lhs =>
      change (x[i] ^ 2 ^ n) * (x[i] ^ 2 ^ n)
    rw [pow_add, pow_one, pow_mul, pow_two]

jax_def (n : ℕ) foo (A):
  i = iota n;
  return A[i]

example (n : ℕ) (x : List ℝ) (h : x.length = n) :
    (Jax.native (foo n)) (.float x) = .float x := by
  have h_cond :  ∀ (a : Fin n), a.val < x.length := fun a ↦ h ▸ a.isLt
  simp [foo, GetElem.getElem, Jax.Impl.idx, iota, Jax.Array.idx]
  simp [Jax.toFin, h_cond]
  congr
  apply List.ext_get
  · simp [h]
  · simp

jax_def (n : ℕ) sum(A):
    return einsum [n] [[#0]] 1 [A]

theorem sum_def (n : ℕ) (x : List ℝ) (hx : x.length = n) (hn : n ≠ 0) :
    (Jax.native (sum n)) (.float x) = .float [x.sum] := by
  simp [sum, Jax.Array.einsum, Jax.List.map?.go, Jax.Array.toTensor, hx, Jax.Array.ofTensor]
  congr 2
  simp [Jax.Tensor.einsum, Jax.Tensor.einprod, Jax.Tensor.unflatten, Fin.mulAdd, Jax.Tensor.flatten,
   Finset.sum_eq_multiset_sum, Multiset.sum, Multiset.foldr, List.sum]
  congr
  apply List.ext_get
  · simp [hx]
  intro _ _ _
  simp
  rfl

jax_def (n : ℕ) normalize_raw(A):
  ans = rep n (sqrt (sum n (A * A)));
  return ans

theorem normalize_raw_def (n : ℕ) (x : List ℝ) (h : x.length = n) (hn : n ≠ 0) :
    Jax.native (normalize_raw n) (.float x) = (.float <| List.replicate n <| Real.sqrt <|
      (x.map (· ^ 2)).sum) := by
  simp [normalize_raw, HMul.hMul]
  have h₁ : (Jax.Array.float x).mul (Jax.Array.float x) = .float (x.map (· ^ 2)) := by
    simp [Jax.Array.mul, Jax.Array.pairwise]
    apply List.ext_get
    · simp
    · simp [pow_two]
  rw [h₁]
  let x' := x.map (· ^ 2)
  have := sum_def n x' (by simp[x',h]) hn
  simp at this
  rw [this]
  simp [Jax.Array.rep]
  congr

jax_def (n : ℕ) Normalize(A):
  norm = normalize_raw n A;
  zero = fillRat n 0;
  one = fillRat n 1;
  norm = Jax.Impl.select (eq norm zero) one norm;
  return A / norm
  
#eval IO.println (Jax.trace (Normalize 10)).outward.code

theorem normalized_def (n : ℕ) (x : List ℝ) (hn : x.length = n) (hn' : n ≠ 0) :
  let out := Jax.native (Normalize n) (.float x);
  ∃ y : List ℝ,
    y.length = n ∧ out = .float y ∧ (y.map (· ^ 2)).sum = 1
  := by
  intro out
  have h_out : out = Jax.native (Normalize n) (.float x) := rfl
  simp [Normalize] at h_out
  have := normalize_raw_def n x hn hn'
  simp at this
  rw [this] at h_out
  simp [Jax.Array.eq] at h_out
  sorry

jax_def (n : ℕ) (m : ℕ) (l : ℕ) matmul(X, Y):
    return einsum [m, n, l] [[#1, #0], [#0, #2]] 1 [X, Y]

theorem matmul_def (n m l : ℕ) (h : n ≠ 0 ∧ m ≠ 0 ∧ l ≠ 0)
  (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    let x' := Jax.Array.ofMatrix x
    let y' := Jax.Array.ofMatrix y
    let z' := Jax.Array.ofMatrix (x * y)
    Jax.native (matmul n m l) x' y' = z' := by
  intro x' y' z'
  simp [matmul, Jax.Array.einsum, Jax.List.map?.go, x', y', z', Jax.Array.ofMatrix]
  congr
  ext i j
  simp [Jax.Tensor.einsum, Jax.Tensor.einprod, show (2 : Fin 3) ≠ 0 by decide, Matrix.mul_apply]
  have h₁ := Finset.sum_apply i Finset.univ fun j i k ↦ x i j * y j k
  conv_lhs =>
    fun
    equals ∑ j, fun k ↦ x i j * y j k =>
      exact h₁
  simp [Finset.sum_apply]

/-

jax_def (n : ℕ) houseHolder_for_antisymm(A):
  def body_fun(i, a):
    i' = rep n i;
    n' = fillInt n n;
    idx = iota n;
    cond = lt idx n' + eq idx n';
    zero = fillRat n 0;
    u = Jax.select cond zero a;
    norm_u = sqrt (einsum [n] [[#0], [#0]] [] [u,u]);
    u = Jax.Impl.setIdx u (i + ofInt [1]) (u[i] - norm_u);
    du = einsum [n, n, n*n] [[#0], [#1], [#1, #2]] [#0, #1] [u, u, a];
    du = du - einsum [n, n] [[#0, #1]] [#1, #0] [du];
    aa = fillInt n 1;
    tmp = (by sorry);
    return (u - du - du)
  return fori_loop (ofInt [n]) A body_fun

#eval IO.println (Jax.trace (houseHolder_for_antisymm 3)).code

/-
The output is
def x0(_x0, x1):
  x2 = x1 * x1
  return x2
def x1(x2, x3):
  x4 = jax.lax.fori_loop(0, x2[0], (lambda i, c: x0(jax.numpy.array([i]), c)), x3)
  return x4
def x0(x2):
  x1 = [3]
  def x3(x12, x10):
    x4 = jax.numpy.arange(3)
    x5 = jax.numpy.zeros(3, dtype=int) + 3
    x6 = x4 < x5
    x7 = x4 == x5
    x8 = x6 + x7
    x9 = jax.numpy.zeros(3, dtype=float) + 0
    x11 = jax.lax.select(x8, x9, x10)
    x13 = [1]
    x14 = x12 + x13
    x15 = x11[x12]
    x16 = jax.numpy.einsum(x11.reshape(3), (0), x11.reshape(3), (0), ())
    x17 = jax.numpy.sqrt(x16)
    x18 = x15 - x17
    x19 = x11.at[x14].set(x18)
    x20 = jax.numpy.einsum(x19.reshape(3), (0), x19.reshape(3), (1), x10.reshape(3, 9), (1, 2), (0, 1))
    x21 = jax.numpy.einsum(x20.reshape(3, 3), (0, 1), (1, 0))
    x22 = x20 - x21
    x23 = x19 - x22
    x24 = x23 - x22
    return x24
  x4 = jax.lax.fori_loop(0, x1[0], (lambda i, c: x3(jax.numpy.array([i]), c)), x2)
  return x4
-/

example (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (h_A_antisymm : ∀ i j, A i j = - A j i) :
    let output := (Jax.native (houseHolder_for_antisymm n)) (Jax.Array.ofMatrix A)
    ∃ B : Matrix (Fin n) (Fin n) ℝ,
      output = Jax.Array.ofMatrix B
      ∧ ∀ i j, B i j = - B j i
      ∧ ∀ i j, i.val + 1 < j.val → B i j = 0
      ∧ ∃ C : Matrix (Fin n) (Fin n) ℝ, B = C * A * C.transpose
      := by
  intro output
  have h_output : output = (Jax.native (houseHolder_for_antisymm n)) (Jax.Array.ofMatrix A) := rfl
  simp at h_output
  unfold houseHolder_for_antisymm at h_output
  sorry
-/
