import JaxProof.Eval

namespace Jax

def FloatAsReal : TensorType → Type
  | ⟨.float, s⟩ => Tensor ℝ s
  | ⟨.int, s⟩ => Tensor ℤ s

def FloatAsReal.zero {σ : TensorType} : FloatAsReal σ :=
  match σ with
  | ⟨.float, _⟩
  | ⟨.int, _⟩ => Tensor.zero

instance (σ : TensorType) : Zero (FloatAsReal σ) := ⟨FloatAsReal.zero⟩

#print Tensor.einprod
#check Tensor.transpose

noncomputable instance : TensorImpl FloatAsReal where
  ofNat i := Int.ofNat i
  impl {args} {out} op := match op with
  | .abs => fun ⟨x, _⟩ => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => x.map abs
  | .acos => fun ⟨x, _⟩ => x.map Real.arccos
  -- `Real.Arcosh` isn't available in this lean version
  | .acosh => fun ⟨x, _⟩ => x.map fun x => Real.log (x + Real.sqrt (x^2 + 1))
  | .add => fun ⟨x, y, _⟩ => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.add x y
  | .mul => fun ⟨x, y, _⟩ => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.map₂ (· * ·) x y
  | .dot_general (α := α) batch contract lhs_indep rhs_indep lhs rhs lhs_inj lhs_surj rhs_inj rhs_surj =>
    fun ⟨x, y, _⟩ =>
    match α with
    | .int => sorry
    | .float =>
      --let lhs_shape := _dot_general.get_shape batch contract lhs_indep lhs
      --let rhs_shape := _dot_general.get_shape batch contract rhs_indep rhs
      let σ : Fin lhs.length → Fin (contract ++ batch ++ lhs_indep).length := fun i ↦
        match lhs[i] with
        | .inl j => Fin.mk j <| by simp; omega
        | .inr (.inl j) => Fin.mk (batch.length + j) <| by simp; omega
        | .inr (.inr j) => Fin.mk (batch.length + contract.length + j) <| by simp; omega
      have σ_inj : Function.Injective σ := by sorry
      have σ_surj : Function.Surjective σ := by sorry
      let σ : Equiv
      --let x' : Tensor ℝ (contract ++ batch ++ lhs_indep) := x.transpose
      sorry
  | _ => 0



end Jax
