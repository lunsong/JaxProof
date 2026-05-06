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

def TensorType.scalar : DType → TensorType := fun α ↦ ⟨α, []⟩

inductive XlaOp : List TensorType → TensorType → Type where
  | abs {σ : TensorType} : XlaOp [σ] σ
  | acos {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | acosh {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | add {σ : TensorType} : XlaOp [σ, σ] σ
  | and {s : Shape} : XlaOp [⟨.int, s⟩, ⟨.int, s⟩] ⟨.int, s⟩
  | argmax {σ : TensorType} (axis : ℕ) : XlaOp [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | argmin {σ : TensorType} (axis : ℕ) : XlaOp [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | asin {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | asinh {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | atan {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | atanh {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i0e {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i1e {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | broadcast {α : DType} (s : List (ℕ × Bool)) :
    XlaOp [⟨α, SSA.Tensor.preBroadcast s⟩] ⟨α, s.map Prod.fst⟩
  | cbrt {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | ceil {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.int, s⟩
  | cholesky {batch : Shape} {n : ℕ} : XlaOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | concat {α : DType} {batch : Shape} {n m : ℕ} {axis : ℕ} :
    XlaOp [⟨α, batch.insertIdx axis n⟩, ⟨α, batch.insertIdx axis m⟩]
      ⟨α, batch.insertIdx axis (n + m)⟩
  | conv {α : DType} {s : Shape} {n m : ℕ} {axis : ℕ} :
    XlaOp [⟨α, s.insertIdx axis n⟩, ⟨α, s.insertIdx axis m⟩] ⟨α, s.insertIdx axis (n + m)⟩
  | convert_type {α β : DType} {s : Shape} : XlaOp [⟨α, s⟩] ⟨β, s⟩
  | cos {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | cosh {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | cumlogsumexp {s : Shape} (axis : ℕ) (reverse : Bool) : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | cummax {σ : TensorType} (axis : ℕ) (reverse : Bool) : XlaOp [σ] σ
  | cummin {σ : TensorType} (axis : ℕ) (reverse : Bool) : XlaOp [σ] σ
  | cumprod {σ : TensorType} (axis : ℕ) (reverse : Bool) : XlaOp [σ] σ
  | cumsum {σ : TensorType} : XlaOp [σ] σ
  | div {σ : TensorType} : XlaOp [σ, σ] σ
  | dot_general {α : DType} (batch contract lhs rhs: List ℕ) : 
    XlaOp [⟨α, contract ++ batch ++ lhs⟩, ⟨α, contract ++ batch ++ rhs⟩] ⟨α, batch ++ lhs ++ rhs⟩
  | sum {α : DType} {s : List ℕ} (n : ℕ) : XlaOp [⟨α, s⟩] ⟨α, s.drop n⟩
  | sorted {α : DType} {batch : List ℕ} {sorted_axes : ℕ} :
    XlaOp [⟨α, batch ++ [sorted_axes]⟩] ⟨α, batch ++ [sorted_axes]⟩
  | transpose {α : DType} {s : List ℕ} (σ : Equiv.Perm (Fin s.length)) :
    XlaOp [⟨α, s⟩] ⟨α, List.ofFn fun i => s.get (σ i)⟩
  | dynamic_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    XlaOp [⟨α, dims.map Prod.fst⟩] ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩
  | dynamic_update_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    XlaOp [⟨α, dims.map Prod.fst⟩, ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩] ⟨α, dims.map Prod.fst⟩
  | eigvals {batch : Shape} {n : ℕ} : XlaOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvalsh {batch : Shape} {n : ℕ} : XlaOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvecs {batch : Shape} {n : ℕ} : XlaOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | eigvecsh {batch : Shape} {n : ℕ} : XlaOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | empty {σ : TensorType} : XlaOp [] σ
  | eq {σ : TensorType} : XlaOp [σ, σ] ⟨.int, σ.shape⟩
  | erf {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | erf_inv {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | erfc {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | exp {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | exp2 {σ : TensorType} : XlaOp [σ] σ
  | expm1 {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  --| fft {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | gather {α : DType} {s s' batch: Shape} :
    XlaOp (⟨α, s⟩ :: List.replicate s.length ⟨.int, s'⟩) ⟨α, s'⟩
  | scatter {α : DType} {s : Shape} {n : ℕ} :
    XlaOp (⟨α, s⟩ :: ⟨α, [n]⟩ :: List.replicate s.length ⟨.int, [n]⟩) ⟨α, s⟩
  | iota {n : ℕ} : XlaOp [] ⟨.int, [n]⟩
  | mul {σ : TensorType} : XlaOp [σ, σ] σ
  | mod {σ : TensorType} : XlaOp [σ, σ] σ
  | div_int {σ : TensorType} : XlaOp [σ, σ] σ
  | zeros {σ : TensorType} : XlaOp [] σ
  | sqrt {s : Shape} : XlaOp [⟨.float, s⟩] ⟨.float, s⟩
  | choice {α : DType} {s : Shape} : XlaOp [⟨.int, s⟩, ⟨α, s⟩, ⟨α, s⟩] ⟨α, s⟩
  | ofNat {σ : TensorType} (val : ℕ) : XlaOp [] σ
  | neg {σ : TensorType} : XlaOp [σ] σ
  | sub {σ : TensorType} : XlaOp [σ, σ] σ
  --| lt : XlaOp (some 2)
  --| select : XlaOp (some 3)
  --| addIdx : XlaOp (some 3)
  --| sin : XlaOp (some 1)
  --| log : XlaOp (some 1)
  --| sqrt : XlaOp (some 1)
  --| einsum (s : List ℕ) : List (List (Fin s.length)) → ℕ → XlaOp none
  --| tuple : XlaOp none
  --| tupleGet : ℕ → XlaOp (some 1)
  --| anonTuple : XlaOp none

inductive XlaHo : SSA.OpType TensorType where
  | repeat {carry aux : List TensorType} :
    XlaHo [⟨carry ++ aux, carry⟩] (TensorType.scalar .int :: carry ++ aux) carry

def XlaOp.toString {args : List TensorType} {out : TensorType} : XlaOp args out → String
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

instance (args : List TensorType) (out : TensorType) : ToString (XlaOp args out) :=
  ⟨XlaOp.toString⟩

end XLA
