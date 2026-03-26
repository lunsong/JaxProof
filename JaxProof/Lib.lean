import JaxProof.Expr
import JaxProof.Eval

namespace Jax

variable {args : List TensorType} {out : TensorType}

def transpose {α : DType} {s : Shape}
  (σ : Equiv.Perm (Fin s.length)) (x : Expr args ⟨α, s⟩) :
    Expr args ⟨α, List.ofFn fun i => s[σ i]⟩ :=
  .bind (.transpose σ) *[x]

def dot_general {α : DType} (batch contract lhs rhs : Shape)
  (x : Expr args ⟨α, contract ++ batch ++ lhs⟩)
  (y : Expr args ⟨α, contract ++ batch ++ rhs⟩) :
    Expr args ⟨α, batch ++ lhs ++ rhs⟩ :=
  .bind (.dot_general batch contract lhs rhs) *[x, y]

instance : Zero (Expr args out) := ⟨.bind .zeros *[]⟩

def iota {n : ℕ} : Expr args ⟨.int, [n]⟩ := .bind .iota *[]

instance (n : ℕ) : OfNat (Expr args out) n := .mk <| .bind (.ofNat n) *[]

instance : Sub (Expr args out) := .mk fun x y => .bind .sub *[x, y]

def cumsum (x : Expr args out) : Expr args out := .bind .cumsum *[x]

@[simp]
theorem DList.get_zero_cons {α : Type} {γ : α → Type} {a : α} {as : List α}
  {x : γ a} {xs : DList γ as} : DList.get 0 (DList.cons x xs) = x := rfl

@[simp]
theorem DList.get_one_cons {α : Type} {γ : α → Type} {a : α} {as : List α}
  {x y : γ a} {xs : DList γ as} : DList.get 1 (DList.cons x (DList.cons y xs)) = y := rfl

def choice {α : DType} {s : Shape} (c : Expr args ⟨.int, s⟩) (x y : Expr args ⟨α, s⟩) :
    Expr args ⟨α, s⟩ :=
  .bind .choice *[c, x, y]

end Jax
