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

def Jax.Array.ofMatrix {n : ℕ} : Matrix (Fin n) (Fin n) ℝ → Array := fun x ↦
  .float <| List.ofFn fun (i : Fin (n * n)) ↦ x i.modNat i.divNat

syntax "#" num : term
macro_rules
  | `(term|# $n) => `(⟨$n, by simp⟩)

jax_def (n : ℕ) sum(A):
    return einsum [n] [[#0]] [] [A]

theorem sum_def (n : ℕ) (x : List ℝ) (hx : x.length = n) (hn : n ≠ 0) :
    (Jax.native (sum n)) (.float x) = .float [x.sum] := by

  simp [sum, Jax.Array.einsum, hn, Jax.allFloat, Jax.allFloat.go, Jax.Einsum.sum, Jax.Einsum.prod,
      Jax.Einsum.prod.go, hx, Jax.ValidShape.size]
  congr
  let shape : Jax.ValidShape := ⟨[n], by simp [hn]⟩
  let idx' (idx : Jax.ValidIdx shape) : ∀ i : Fin 1, Fin [n][i.val] :=
    fun i ↦ Fin.mk (idx [(0 : Fin 1)][i.val]).val <| by
      have := (idx [(0 : Fin 1)][i.val]).isLt
      simpa
  have h_idx' (idx : Jax.ValidIdx shape) : idx' idx = idx := by
    ext i
    have : i = 0 := by omega
    rw[this]
    simp [idx']
  simp [idx'] at h_idx'
  conv =>
    lhs; arg 2; intro idx
    conv =>
      arg 2
      rw[h_idx' idx]
  have h_sum : x.sum = ∑ i : Fin x.length, x[i] := Eq.symm (Fin.sum_univ_getElem x)
  rw [h_sum]
  refine Function.Bijective.sum_comp ?_ x.get
  constructor
  · intro i j
    simp
    intro h
    rw [Fin.val_eq_val] at h
    exact (Jax.flattenEuiv shape).injective h
  · intro i
    have ⟨a, ha⟩ := (Jax.flattenEuiv shape).surjective <| i.cast <| by
      simp[shape, Jax.ValidShape.size, hx]
    use a
    simp
    simpa [← Fin.val_eq_val] using ha

jax_def (n : ℕ) normalize(A):
  norm = sqrt (sum n (A * A));
  return A / rep n norm
  
#eval IO.println (Jax.trace (normalize 10)).outward.code

theorem normalized_def (n : ℕ) (x : List ℝ) (hn : x.length = n) (hn' : n ≠ 0) :
  let out := (Jax.native (normalize n)) (.float x);
  ∃ y : List ℝ,
    y.length = n ∧ out = .float y ∧ (y.map (· ^ 2)).sum = 1
  := by
  intro out
  have h_out : out = (Jax.native (normalize n)) (.float x) := rfl
  simp [normalize, HMul.hMul] at h_out
  have h₁ : (Jax.Array.float x).mul (Jax.Array.float x) = .float (x.map (· ^ 2)) := by
    simp [Jax.Array.mul, Jax.Array.pairwise]
    apply List.ext_get
    · simp
    · simp [pow_two]
  have h₂ : (x.map (· ^ 2)).length = n := by simp[hn]
  have h₃ := sum_def n _ h₂ hn' 
  simp at h₃
  conv at h_out =>
    rhs; arg 2; arg 2; arg 2
    rw [h₁, h₃]
  simp [Jax.Array.rep, HDiv.hDiv, hn, hn', Jax.Array.div] at h_out


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
