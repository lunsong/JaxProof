import JaxProof.Expr

namespace Jax

def transpose {args : List TensorType} {α : DType} {s : Shape}
  (σ : Equiv.Perm (Fin s.length)) (x : Expr args ⟨α, s⟩) :
    Expr args ⟨α, List.ofFn fun i => s[σ i]⟩ :=
  .bind (.transpose σ) *[x]

def dot_general {args : List TensorType} {α : DType} (batch contract lhs rhs : Shape)
  (x : Expr args ⟨α, contract ++ batch ++ lhs⟩)
  (y : Expr args ⟨α, contract ++ batch ++ rhs⟩) :
    Expr args ⟨α, batch ++ lhs ++ rhs⟩ :=
  .bind (.dot_general batch contract lhs rhs) *[x, y]

instance (args : List TensorType) (out : TensorType) : Zero (Expr args out) := ⟨.bind .zeros *[]⟩

def iota {args : List TensorType} {n : ℕ} : Expr args ⟨.int, [n]⟩ := .bind .iota *[]

end Jax
