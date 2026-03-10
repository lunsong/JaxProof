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
  | .dot_general batch contract lhs_indep rhs_indep lhs rhs lhs_inj lhs_surj rhs_inj rhs_surj =>
    sorry
  | _ => 0



end Jax
