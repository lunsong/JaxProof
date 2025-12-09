import JaxProof.Api
import Mathlib.Data.Matrix.Mul

open Jax.Impl

-- define the jax function
jax_def f(n, x):
  def bodyfn(i, a):
    return a * a
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

-- prove mathematical properties
example (n : ℕ) (x : List ℝ) :
    (Jax.native f) (.int [n]) (.float x) = .float (x.map (· ^ (2 ^ n))) := by
  simp[f]
  induction n with
  | zero => simp
  | succ n ih =>
    simp [ih]
    simp [HMul.hMul, Jax.withLift₂, Jax.Array.pairwise, Jax.Array.mul]
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

def Jax.Array.ofMatrix {n : ℕ} : Matrix (Fin n) (Fin n) ℝ → Array := fun x ↦
  .float <| List.ofFn fun (i : Fin (n * n)) ↦ x i.modNat i.divNat

syntax "#" num : term
macro_rules
  | `(term|# $n) => `(⟨$n, by simp⟩)

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
      := by sorry
