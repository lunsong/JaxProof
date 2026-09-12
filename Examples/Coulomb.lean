import Xla

def diag_mask {n : ℕ} : Xla.SimpleExpr [] ⟨.int, [n, n]⟩ :=
  let idx := Xla.iota n;
  let i := Xla.broadcast [⟨n, true⟩, ⟨n, false⟩] idx;
  let j := Xla.broadcast [⟨n, false⟩, ⟨n, true⟩] idx;
  Xla.eq i j

theorem diag_mask_def {n : ℕ} (i j : Fin n) :
    diag_mask.eval i j = if i = j then (1 : ℤ) else 0 := by
  simp [reduce_soir, reduce_xla, diag_mask, reduce_tensor]
  congr 1
  apply propext
  constructor
  · intro h
    grind
  · intro h; simp[h, Soir.Index.single, id]

def mutual_displacement {n_atom : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n_atom, 3]⟩]
    ⟨.float, [n_atom, n_atom, 3]⟩ :=
  Soir.Expr.ofFn fun x =>
    let sender   := Xla.broadcast [⟨n_atom,  true⟩, ⟨n_atom, false⟩, ⟨3, true⟩] x;
    let receiver := Xla.broadcast [⟨n_atom, false⟩, ⟨n_atom,  true⟩, ⟨3, true⟩] x;
    Xla.sub sender receiver

def mutual_distance {n_atom : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n_atom, 3]⟩]
    ⟨.float, [n_atom, n_atom]⟩ :=
  Soir.Expr.ofFn fun x =>
    let x := mutual_displacement.apply x
    let x2 := Xla.mul x x;
    let x2 := Xla.transpose x2 perm[2,0,1];
    Xla.sum 1 x2

theorem mutual_distance_def {n_atom}
  (x : Xla.Tensor ℝ [n_atom, 3])
  (n m : Fin n_atom) :
  mutual_distance.eval x n m = ∑ i, (x n i - x m i) ^ 2 := by
  simp [mutual_distance, mutual_displacement, reduce_xla, reduce_soir, reduce_tensor, ← pow_two]
  conv_lhs =>
    change (Xla.Tensor.sumN (s := [3, n_atom, n_atom]) 1 (fun i a b ↦ (x a i - x b i)^2)) n m
    fun
    simp [Xla.Tensor.sumN]
  erw [Finset.sum_apply, Finset.sum_apply]

def coulomb {n_atom : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n_atom, 3]⟩]
    ⟨.float, []⟩ :=
  Soir.Expr.ofFn fun x ↦
    let r2 := mutual_distance.apply x;
    let d := Xla.div (Xla.ofNat 1) (Xla.sqrt r2);
    let mask := diag_mask.apply .nil;
    let d := Xla.choice mask (Xla.ofNat 0) d;
    Xla.sum 2 d

#eval IO.println (coulomb (n_atom := 12)).code

open Xla in
example {n_atom : ℕ} (x : Fin n_atom → Fin 3 → ℝ) :
    coulomb.eval x = ∑ i, ∑ j with i ≠ j, 1 / √(∑ k, (x i k - x j k) ^ 2) := by
  have h0 := diag_mask_def (n := n_atom)
  have h1 := mutual_distance_def (n_atom := n_atom)
  simp [Xla.SimpleExpr.eval, Soir.Curry.map] at h0 h1
  replace h0 := fun i ↦ funext (h0 i)
  replace h0 := funext h0
  conv_lhs =>
    simp [coulomb, reduce_xla, reduce_soir, h0, h1, reduce_tensor]
    arg 2; intro i
    erw [Finset.sum_apply]
    simp [Finset.sum_ite]
  congr
  ext i
  congr <;> grind
