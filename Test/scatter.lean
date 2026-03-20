import JaxProof

def inv {n : ℕ} :=
  xla with
    x : int [n]
  returns
    int [n]
  begin
    return .bind .scatter *[0, x, Jax.iota]

#eval IO.println (inv (n := 10)).code

theorem inv_def (n : ℕ) (σ : Equiv.Perm (Fin n)) :
    let x : Jax.FloatAsReal ⟨.int, [n]⟩ := (Int.ofNat ∘ Fin.val ∘ σ);
    let y : Jax.FloatAsReal ⟨.int, [n]⟩ := (Int.ofNat ∘ Fin.val ∘ σ.symm);
    inv.eval _ *[x] = *[y] :=
  match n with
  | 0 => by
    intro x y
    simp [Jax.ExprGroup.eval, inv]
    ext i
    nomatch i
  | n + 1 => by
    intro x y
    simp [inv, Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval]
    let ι : Jax.FloatAsReal ⟨.int, [n + 1]⟩ := Jax.Expr.eval Jax.FloatAsReal (Jax.DList.cons x Jax.DList.nil) Jax.iota
    change Jax.FloatAsReal.scatter (s := [n + 1]) 0 x *[ι] = y
    --have : σ ∈ (⊤ : Submonoid (Equiv.Perm (Fin (n + 1)))) := by simp
    --rw [← Equiv.Perm.mclosure_swap_castSucc_succ] at this
    --refine Submonoid.closure_induction  ?_ ?_ ?_ this
    simp [Jax.FloatAsReal.scatter]
    conv_lhs =>
      arg 1; intro i r
      arg 1
      simp [Jax.FloatAsReal.scatter.get_indices]
      rw [show r = 0 by omega]
      simp

    let fn {n : ℕ} (x ι : Fin (n + 1) → ℤ) : Fin (n + 1) → ℤ :=
      let ι' : Fin (n + 1) → Jax.ValidIdx [n + 1] := fun i r =>
        have : NeZero [n + 1][r.val] := .mk <| by simp
        Fin.intCast (ι i)
      Jax.FloatAsReal.scatter.update ι' 0 x
    change fn x ι = y

    let z : Fin (n + 1) → ℤ := fun i => σ.symm i

    suffices h1 : ∀ σ : Equiv.Perm (Fin (n + 1)), ∀ (x ι : Fin (n + 1) → ℤ),
      fn (x ∘ σ) (ι ∘ σ) = fn x ι by
      specialize h1 σ.symm x ι
      conv_lhs at h1 =>
        conv =>
          arg 1
          change fun i => σ (σ.symm i)
          simp
        arg 2
        change fun i => σ.symm i
      rw [← h1]
      suffices h2 : ∀ n, ∀ x : Fin (n + 1) → ℤ, fn (fun i => i) x = x by
        rw [h2]
        rfl
      intro n
      induction n with
      | zero =>
        intro x
        simp [fn]
        ext i
    sorry
    --have h1 (σ : Equiv.Perm (Fin (n + 1))) (x ι : Fin (n + 1) → ℤ) :
    --  fn (x ∘ σ) (ι ∘ σ) = (fn x ι) ∘ σ := sorry



