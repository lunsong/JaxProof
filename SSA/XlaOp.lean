import SSA.Core
import SSA.Tensor

namespace XLA

inductive DType : Type where
  | int : DType
  | float : DType

instance : ToString DType where
  toString
  | .int => "int"
  | .float => "float"

abbrev Shape : Type := List ℕ

structure TensorType where
  dtype : DType
  shape : Shape

inductive Op : List TensorType → TensorType → Type where
  | abs {σ : TensorType} : Op [σ] σ
  | acos {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | acosh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | add {σ : TensorType} : Op [σ, σ] σ
  | and {s : Shape} : Op [⟨.int, s⟩, ⟨.int, s⟩] ⟨.int, s⟩
  | argmax {σ : TensorType} (axis : ℕ) : Op [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | argmin {σ : TensorType} (axis : ℕ) : Op [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | asin {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | asinh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | atan {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | atanh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i0e {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i1e {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | broadcast {α : DType} (s : List (ℕ × Bool)) :
    Op [⟨α, SSA.Tensor.preBroadcast s⟩] ⟨α, s.map Prod.fst⟩
  | cbrt {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | ceil {s : Shape} : Op [⟨.float, s⟩] ⟨.int, s⟩
  | cholesky {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | concat {α : DType} {batch : Shape} {n m : ℕ} {axis : ℕ} :
    Op [⟨α, batch.insertIdx axis n⟩, ⟨α, batch.insertIdx axis m⟩] ⟨α, batch.insertIdx axis (n + m)⟩
  | conv {α : DType} {s : Shape} {n m : ℕ} {axis : ℕ} :
    Op [⟨α, s.insertIdx axis n⟩, ⟨α, s.insertIdx axis m⟩] ⟨α, s.insertIdx axis (n + m)⟩
  | convert_type {α β : DType} {s : Shape} : Op [⟨α, s⟩] ⟨β, s⟩
  | cos {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | cosh {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | cumlogsumexp {s : Shape} (axis : ℕ) (reverse : Bool) : Op [⟨.float, s⟩] ⟨.float, s⟩
  | cummax {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | cummin {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | cumprod {σ : TensorType} (axis : ℕ) (reverse : Bool) : Op [σ] σ
  | cumsum {σ : TensorType} : Op [σ] σ
  | div {σ : TensorType} : Op [σ, σ] σ
  | dot_general {α : DType} (batch contract lhs rhs: List ℕ) : 
    Op [⟨α, contract ++ batch ++ lhs⟩, ⟨α, contract ++ batch ++ rhs⟩] ⟨α, batch ++ lhs ++ rhs⟩
  | sum {α : DType} {s : List ℕ} (n : ℕ) : Op [⟨α, s⟩] ⟨α, s.drop n⟩
  | sorted {α : DType} {batch : List ℕ} {sorted_axes : ℕ} :
    Op [⟨α, batch ++ [sorted_axes]⟩] ⟨α, batch ++ [sorted_axes]⟩
  | transpose {α : DType} {s : List ℕ} (σ : Equiv.Perm (Fin s.length)) :
    Op [⟨α, s⟩] ⟨α, List.ofFn fun i => s.get (σ i)⟩
  | dynamic_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    Op [⟨α, dims.map Prod.fst⟩] ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩
  | dynamic_update_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    Op [⟨α, dims.map Prod.fst⟩, ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩] ⟨α, dims.map Prod.fst⟩
  | eigvals {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvalsh {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvecs {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | eigvecsh {batch : Shape} {n : ℕ} : Op [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | empty {σ : TensorType} : Op [] σ
  | eq {σ : TensorType} : Op [σ, σ] ⟨.int, σ.shape⟩
  | erf {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | erf_inv {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | erfc {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | exp {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | exp2 {σ : TensorType} : Op [σ] σ
  | expm1 {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  --| fft {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | gather {α : DType} {s s' batch: Shape} :
    Op (⟨α, s⟩ :: List.replicate s.length ⟨.int, s'⟩) ⟨α, s'⟩
  | scatter {α : DType} {s : Shape} {n : ℕ} :
    Op (⟨α, s⟩ :: ⟨α, [n]⟩ :: List.replicate s.length ⟨.int, [n]⟩) ⟨α, s⟩
  | iota {n : ℕ} : Op [] ⟨.int, [n]⟩
  | mul {σ : TensorType} : Op [σ, σ] σ
  | mod {σ : TensorType} : Op [σ, σ] σ
  | div_int {σ : TensorType} : Op [σ, σ] σ
  | zeros {σ : TensorType} : Op [] σ
  | sqrt {s : Shape} : Op [⟨.float, s⟩] ⟨.float, s⟩
  | choice {α : DType} {s : Shape} : Op [⟨.int, s⟩, ⟨α, s⟩, ⟨α, s⟩] ⟨α, s⟩
  | ofNat {σ : TensorType} (val : ℕ) : Op [] σ
  | neg {σ : TensorType} : Op [σ] σ
  | sub {σ : TensorType} : Op [σ, σ] σ
  --| lt : Op (some 2)
  --| select : Op (some 3)
  --| addIdx : Op (some 3)
  --| sin : Op (some 1)
  --| log : Op (some 1)
  --| sqrt : Op (some 1)
  --| einsum (s : List ℕ) : List (List (Fin s.length)) → ℕ → Op none
  --| tuple : Op none
  --| tupleGet : ℕ → Op (some 1)
  --| anonTuple : Op none

def Op.toString {args : List TensorType} {out : TensorType} : Op args out → String
  | add => "add"
  | cos => "cos"
  | concat => "concat"
  | mul => "mul"
  | transpose σ =>
    let σ : List ℕ := List.ofFn fun i => σ i
    s!"transpose {σ}"
  | dot_general batch contract lhs rhs => s!"dot_general {contract.length} {batch.length}"
  | scatter => "scatter"
  | zeros (σ := σ) => s!"zeros {σ.dtype} {σ.shape}"
  | iota (n := n) => s!"iota {n}"
  | sum n => s!"sum {n}"
  | broadcast s => s!"braodcast {s.map Prod.snd}"
  | div => "div"
  | sqrt => "sqrt"
  | cumsum => "cumsum"
  | choice => "where"
  | ofNat (σ := ⟨α, s⟩) n => s!"const {α} {s} {n}"
  | neg => "neg"
  | sub => "sub"
  | _ => "unimplemented"

instance (args : List TensorType) (out : TensorType) : ToString (Op args out) :=
  ⟨Op.toString⟩

end XLA
