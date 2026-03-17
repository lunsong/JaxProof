import JaxProof.Expr

namespace Jax

def transpose {args : List Jax.TensorType} {α : Jax.DType} {s : List ℕ}
  (σ : Equiv.Perm (Fin s.length)) (x : Jax.Expr args ⟨α, s⟩) :
    Jax.Expr args ⟨α, List.ofFn fun i => s[σ i]⟩ :=
  .bind (.transpose σ) *[x]

def dot_general {args : List TensorType} {α : DType} (batch contract lhs rhs : List ℕ)
  (x : Expr args ⟨α, contract ++ batch ++ lhs⟩)
  (y : Expr args ⟨α, contract ++ batch ++ rhs⟩) :
    Expr args ⟨α, batch ++ lhs ++ rhs⟩ :=
  .bind (.dot_general batch contract lhs rhs) *[x, y]

end Jax
