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
    simp [inv, Jax.ExprGroup.eval]
    ext i
    nomatch i
  | 1 => by
    intro x y
    simp [inv, Jax.ExprGroup.eval]
    ext i
    cases (show i = 0 by omega)
    simp [Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval, Jax.FloatAsReal.scatter,
      Jax.FloatAsReal.scatter.update, Jax.Tensor.of, Jax.Tensor.get, Function.update]
    split
    · change x 0 = y 0
      simp [x, y]
    · rename_i h
      contrapose! h
      apply funext
      intro i
      match i with
      | 0 => rfl
  | n + 2 =>
    let pivot : Fin (n + 1) := if h : σ 0 = 0 then 0 else (σ 0).pred h
    let f : Fin (n + 1) → Fin (n + 1) := fun i => Fin.predAbove pivot (σ i.succ)
    have helper (i : Fin (n + 1)) : σ i.succ ≠ σ 0 := by
      intro h
      have := σ.injective h
      simp at this
    let f_inv : Fin (n + 1) → Fin (n + 1) := fun i =>
      let j := σ.symm (Fin.succAbove (σ 0) i)
      j.pred <| by
        simp only [j]
        nth_rw 2 [show 0 = σ.symm (σ 0) by simp]
        intro h
        have := σ.symm.injective h
        simp at this
    let σ' : Equiv.Perm (Fin (n + 1)) := {
      toFun := f
      invFun := f_inv
      left_inv x := by
        simp [f, f_inv, pivot]
        split_ifs with h
        · simp[h, h ▸ helper x]
        · conv_lhs =>
            arg 1; arg 2; arg 1
            equals ((σ 0).pred h).succ => simp
          simp only [Fin.succ_succAbove_predAbove (p := (σ 0).pred h) (i := σ x.succ) (by simp)]
          simp
      right_inv x := by
        simp [f, f_inv, pivot]
        split_ifs with h
        · simp [h]
        · simp [Fin.pred, Fin.predAbove, Fin.succAbove]
          split_ifs with h1 h2 <;> omega


    }
    sorry

        

      

    
--  intro x y
--  simp [inv, Jax.ExprGroup.eval, Jax.Expr.eval, Jax.TensorImpl.impl, Jax.Expr.eval.recursive_eval]
--  let ι : Jax.FloatAsReal ⟨.int, [n]⟩ := Jax.Expr.eval Jax.FloatAsReal (Jax.DList.cons x Jax.DList.nil) Jax.iota
--  change Jax.FloatAsReal.scatter (s := [n]) 0 x *[ι] = y
--  revert σ 
--  induction n with
--  | zero =>
--    intro σ x y ι
--    simp [Jax.FloatAsReal.scatter]
--    ext i
--    nomatch i
--
--  | succ n ih =>
--    intro σ x y ι
--    simp [Jax.FloatAsReal.scatter]
--    conv_lhs =>
--      arg 1; intro i r; arg 1
--      rw [show r = 1 by omega]
--      change i.val
--    simp [Jax.FloatAsReal.scatter.update]
--    conv at ih =>
--      intro σ x y ι
--      simp [Jax.FloatAsReal.scatter]


      




