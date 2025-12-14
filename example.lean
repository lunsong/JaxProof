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

def ofMatrix {n m : ℕ} : Matrix (Fin n) (Fin m) ℝ → Jax.Array := fun x ↦
  .float <| List.ofFn fun (i : Fin (n * m)) ↦ x i.divNat i.modNat

macro "#" noWs n:num : term => `(⟨$n, by simp +decide⟩)

jax_def (n : ℕ) sum(A):
    return einsum [n] [[#0]] [] [A]

theorem sum_def (n : ℕ) (x : List ℝ) (hx : x.length = n) (hn : n ≠ 0) :
    (Jax.native (sum n)) (.float x) = .float [x.sum] := by

  simp [sum, Jax.Array.einsum, hn, Jax.allFloat, Jax.allFloat.go, Jax.Einsum.sum, Jax.Einsum.prod,
      Jax.Einsum.prod.go, hx]
  congr
  conv =>
    lhs; arg 2; intro idx; arg 2
    equals idx.flatten =>
      congr
      funext i
      have : i = 0 := by omega
      simp [← Fin.val_eq_val]
      rw[this]
      congr
  have h_sum : x.sum = ∑ i : Fin x.length, x[i] := Eq.symm (Fin.sum_univ_getElem x)
  rw [h_sum]
  refine Function.Bijective.sum_comp ?_ x.get
  constructor
  · intro i j
    simp
    intro h
    rw [Fin.val_eq_val] at h
    exact (Jax.ValidIdx.equiv [n]).injective h
  · intro i
    have ⟨a, ha⟩ := (Jax.ValidIdx.equiv [n]).surjective <| i.cast <| by simp[hx]
    use a
    simp
    simpa [← Fin.val_eq_val] using ha


jax_def (n : ℕ) Normalize(A):
  norm = sqrt (sum n (A * A));
  return A / rep n norm
  
#eval IO.println (Jax.trace (Normalize 10)).outward.code

theorem normalized_def (n : ℕ) (x : List ℝ) (hn : x.length = n) (hn' : n ≠ 0) :
  let out := Jax.native (Normalize n) (.float x);
  ∃ y : List ℝ,
    y.length = n ∧ out = .float y ∧ (y.map (· ^ 2)).sum = 1
  := by
  intro out
  have h_out : out = Jax.native (Normalize n) (.float x) := rfl
  simp [Normalize, HMul.hMul] at h_out
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
  sorry

jax_def (n : ℕ) (m : ℕ) (l : ℕ) matmul(X, Y):
    return einsum [n, m, l] [[#0, #1], [#1, #2]] [#0, #2] [X, Y]

theorem matmul_def (n m l : ℕ) (h : n ≠ 0 ∧ m ≠ 0 ∧ l ≠ 0)
  (x : Matrix (Fin n) (Fin m) ℝ) (y : Matrix (Fin m) (Fin l) ℝ) :
    Jax.native (matmul n m l) (ofMatrix x) (ofMatrix y) = ofMatrix (x * y) := by
  simp[matmul, Jax.Array.einsum, h, Jax.allFloat, Jax.allFloat.go, ofMatrix, Jax.Einsum.sum,
    Jax.Einsum.prod, Jax.Einsum.prod.go]
  congr 1
  apply List.ext_get
  · simp
  simp [Jax.ValidIdx.unflatten, List.getElem_ofFn]
  intro a ha _
  let i : Fin n := Fin.divNat ⟨a, ha⟩
  let j : Fin l := Fin.modNat ⟨a, ha⟩
  --let idx_shape : Jax.ValidShape := ⟨[n,m,l], by simp[h]⟩
  have : NeZero [n,m,l].length := ⟨by simp⟩
  conv =>
    lhs
    conv =>
      arg 1; arg 1; intro idx
      conv =>
        arg 1
        rw [← Fin.val_eq_val]
        simp
        change (idx 0).val = i.val
        rw [Fin.val_eq_val]
      arg 2
      rw [← Fin.val_eq_val]
      simp
      change (idx 2).val = j.val
      rw [Fin.val_eq_val]
    arg 2; intro idx
    conv =>
      arg 1
      simp only [GetElem.getElem, List.get_ofFn, Fin.modNat, Fin.divNat]
      simp[Jax.ValidIdx.flatten]
      conv =>
        arg 1
        equals idx 0 =>
          rw [← Fin.val_eq_val]
          have : 0 < m := Nat.zero_lt_of_lt (idx 1).isLt
          simp
          rw [Nat.add_div, Nat.mul_div_cancel,
            show (idx 1).val / m = 0 from Nat.div_eq_zero_iff.mpr (.inr (idx 1).isLt)]
          simp
          exact Nat.mod_lt _ this
          exact this
          exact this
      arg 2
      equals idx 1 =>
        rw [← Fin.val_eq_val]
        simp
        exact Nat.mod_eq_of_lt (idx 1).isLt
    arg 2
    simp only [GetElem.getElem, List.get_ofFn, Fin.modNat, Fin.divNat]
    simp [Jax.ValidIdx.flatten]
    conv =>
      arg 2
      equals idx 2 =>
        rw [← Fin.val_eq_val]
        simp
        exact Nat.mod_eq_of_lt (idx 2).isLt
    arg 1
    equals idx 1 =>
      rw [← Fin.val_eq_val]
      simp
      have h₁ : idx 0 < n := (idx 0).isLt
      have h₂ : idx 1 < m := (idx 1).isLt
      rw [Nat.add_div (by omega), Nat.mul_div_cancel _ (by omega),
        show (idx 2).val / l = 0 from Nat.div_eq_zero_iff.mpr (.inr (idx 2).isLt), add_zero]
      simp
      exact (Nat.mod_lt _ (by omega))
  conv_rhs =>
    change (x * y) i j
  rw [Matrix.mul_apply]
  apply Finset.sum_nbij (fun idx ↦ idx 1) 
  · simp
  · intro x hx y hy h
    simp at hx hy h
    simp [Jax.ValidIdx] at x y
    funext k
    fin_cases k
    · exact hx.1.trans hy.1.symm
    · exact h
    · exact hx.2.trans hy.2.symm
  · intro x hx
    simp
    let x' : Jax.ValidIdx [n,m,l] := fun a ↦
      if h₁ : a = 0 then 
        i.cast <| by simp[ h₁]
      else if h₂ : a = 1 then
        x.cast <| by simp[ h₂]
      else
        j.cast <| by
          have := a.isLt
          simp only [List.length_cons, List.length_nil, Nat.reduceAdd, zero_add] at this
          have : a = 2 := by
            rw [← Fin.val_eq_val] at h₁ h₂ ⊢
            simp[-Fin.val_eq_zero_iff] at h₁ h₂ ⊢; 
            generalize a.val = a' at h₁ h₂ this ⊢
            omega
          simp[this]
    use x'
    simp +decide [x',show (2 : Fin 3) ≠ 0 by decide, show (2 : Fin 3) ≠ 1 by decide]
  simp +contextual


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
