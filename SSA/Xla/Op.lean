import SSA.Core
import SSA.Tensor

namespace Xla

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

inductive XlaPrimOp : List TensorType → TensorType → Type where
  | abs {σ : TensorType} : XlaPrimOp [σ] σ
  | acos {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | acosh {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | add {σ : TensorType} : XlaPrimOp [σ, σ] σ
  | and {s : Shape} : XlaPrimOp [⟨.int, s⟩, ⟨.int, s⟩] ⟨.int, s⟩
  | argmax {σ : TensorType} (axis : ℕ) : XlaPrimOp [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | argmin {σ : TensorType} (axis : ℕ) : XlaPrimOp [σ] ⟨.int, σ.shape.eraseIdx axis⟩ 
  | asin {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | asinh {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | atan {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | atanh {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i0e {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | bessel_i1e {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | broadcast {α : DType} (s : List (ℕ × Bool)) :
    XlaPrimOp [⟨α, SSA.Tensor.preBroadcast s⟩] ⟨α, s.map Prod.fst⟩
  | cbrt {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | ceil {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.int, s⟩
  | cholesky {batch : Shape} {n : ℕ} :
    XlaPrimOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | concat {α : DType} {batch : Shape} {n m : ℕ} {axis : ℕ} :
    XlaPrimOp [⟨α, batch.insertIdx axis n⟩, ⟨α, batch.insertIdx axis m⟩]
      ⟨α, batch.insertIdx axis (n + m)⟩
  | conv {α : DType} {s : Shape} {n m : ℕ} {axis : ℕ} :
    XlaPrimOp [⟨α, s.insertIdx axis n⟩, ⟨α, s.insertIdx axis m⟩] ⟨α, s.insertIdx axis (n + m)⟩
  | convert_type {α β : DType} {s : Shape} : XlaPrimOp [⟨α, s⟩] ⟨β, s⟩
  | cos {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | cosh {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | cumlogsumexp {s : Shape} (axis : ℕ) (reverse : Bool) : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | cummax {σ : TensorType} (axis : ℕ) (reverse : Bool) : XlaPrimOp [σ] σ
  | cummin {σ : TensorType} (axis : ℕ) (reverse : Bool) : XlaPrimOp [σ] σ
  | cumprod {σ : TensorType} (axis : ℕ) (reverse : Bool) : XlaPrimOp [σ] σ
  | cumsum {σ : TensorType} : XlaPrimOp [σ] σ
  | div {σ : TensorType} : XlaPrimOp [σ, σ] σ
  | dot_general {α : DType} (batch contract lhs rhs: List ℕ) : 
    XlaPrimOp
      [⟨α, contract ++ batch ++ lhs⟩, ⟨α, contract ++ batch ++ rhs⟩] ⟨α, batch ++ lhs ++ rhs⟩
  | sum {α : DType} {s : List ℕ} (n : ℕ) : XlaPrimOp [⟨α, s⟩] ⟨α, s.drop n⟩
  | sorted {α : DType} {batch : List ℕ} {sorted_axes : ℕ} :
    XlaPrimOp [⟨α, batch ++ [sorted_axes]⟩] ⟨α, batch ++ [sorted_axes]⟩
  | transpose {α : DType} {s : List ℕ} (σ : Equiv.Perm (Fin s.length)) :
    XlaPrimOp [⟨α, s⟩] ⟨α, List.ofFn fun i => s.get (σ i)⟩
  | dynamic_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    XlaPrimOp [⟨α, dims.map Prod.fst⟩] ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩
  | dynamic_update_slice {α : DType} (dims : List (ℕ × ℕ × ℕ)) :
    XlaPrimOp [⟨α, dims.map Prod.fst⟩, ⟨α, dims.map (Prod.snd ∘ Prod.snd)⟩] ⟨α, dims.map Prod.fst⟩
  | eigvals {batch : Shape} {n : ℕ} : XlaPrimOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvalsh {batch : Shape} {n : ℕ} : XlaPrimOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n]⟩
  | eigvecs {batch : Shape} {n : ℕ} :
    XlaPrimOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | eigvecsh {batch : Shape} {n : ℕ} :
    XlaPrimOp [⟨.float, batch ++ [n, n]⟩] ⟨.float, batch ++ [n, n]⟩
  | empty {σ : TensorType} : XlaPrimOp [] σ
  | eq {σ : TensorType} : XlaPrimOp [σ, σ] ⟨.int, σ.shape⟩
  | erf {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | erf_inv {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | erfc {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | exp {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | exp2 {σ : TensorType} : XlaPrimOp [σ] σ
  | expm1 {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | log {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | sin {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  --| fft {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | gather {α : DType} {s s' : Shape} :
    XlaPrimOp (⟨α, s⟩ :: List.replicate s.length ⟨.int, s'⟩) ⟨α, s'⟩
  | scatter {α : DType} {s : Shape} {n : ℕ} :
    XlaPrimOp (⟨α, s⟩ :: ⟨α, [n]⟩ :: List.replicate s.length ⟨.int, [n]⟩) ⟨α, s⟩
  | iota {n : ℕ} : XlaPrimOp [] ⟨.int, [n]⟩
  | mul {σ : TensorType} : XlaPrimOp [σ, σ] σ
  | mod {σ : TensorType} : XlaPrimOp [σ, σ] σ
  | div_int {σ : TensorType} : XlaPrimOp [σ, σ] σ
  | zeros {σ : TensorType} : XlaPrimOp [] σ
  | sqrt {s : Shape} : XlaPrimOp [⟨.float, s⟩] ⟨.float, s⟩
  | choice {α : DType} {s : Shape} : XlaPrimOp [⟨.int, s⟩, ⟨α, s⟩, ⟨α, s⟩] ⟨α, s⟩
  | ofNat {σ : TensorType} (val : ℕ) : XlaPrimOp [] σ
  | neg {σ : TensorType} : XlaPrimOp [σ] σ
  | sub {σ : TensorType} : XlaPrimOp [σ, σ] σ
  | einsum (s : Shape) (indices : List (List (Fin s.length))) (n : ℕ) :
    XlaPrimOp (indices.map fun i ↦ ⟨.float, i.map s.get⟩) ⟨.float, s.drop n⟩
  | flatten {α : DType} {s : Shape} : XlaPrimOp [⟨α, s⟩] ⟨α, [s.prod]⟩
  | unflatten {α : DType} (s : Shape) : XlaPrimOp [⟨α, [s.prod]⟩] ⟨α, s⟩
  | cast {α : DType} {s s' : Shape} : s = s' → XlaPrimOp [⟨α, s⟩] ⟨α, s'⟩
  --| lt : XlaPrimOp (some 2)
  --| select : XlaPrimOp (some 3)
  --| addIdx : XlaPrimOp (some 3)
  --| lt : XlaPrimOp (some 2)
  --| select : XlaPrimOp (some 3)
  --| addIdx : XlaPrimOp (some 3)
  --| einsum (s : List ℕ) : List (List (Fin s.length)) → ℕ → XlaPrimOp none
  --| tuple : XlaPrimOp none
  --| tupleGet : ℕ → XlaPrimOp (some 1)
  --| anonTuple : XlaPrimOp none

inductive XlaHigherOp : SSA.OpType TensorType where
  | repeat {carry aux : List TensorType} :
    XlaHigherOp [⟨carry ++ aux, carry⟩] (TensorType.scalar .int :: carry ++ aux) carry
  | vmap {args aux outs : List TensorType} {batch : ℕ} :
    XlaHigherOp [⟨args ++ aux, outs⟩]
      ((args.map fun ⟨α, s⟩ ↦ ⟨α, batch :: s⟩) ++ aux)
      (outs.map fun ⟨α, s⟩ ↦ ⟨α, batch :: s⟩)

def XlaPrimOp.toString {args : List TensorType} {out : TensorType} : XlaPrimOp args out → String
  | abs => "abs"
  | acos => "acos"
  | acosh => "acosh"
  | add => "add"
  | and => "and"
  | argmax axis => s!"argmax {axis}"
  | argmin axis => s!"argmin {axis}"
  | asin => "asin"
  | asinh => "asinh"
  | atan => "atan"
  | atanh => "atanh"
  | bessel_i0e => "bessel_i0e"
  | bessel_i1e => "bessel_i1e"
  | broadcast s => s!"broadcast {s.map Prod.snd}"
  | cbrt => "cbrt"
  | ceil => "ceil"
  | cholesky => "cholesky"
  | concat => "concat"
  | conv => "conv"
  | convert_type => "convert_type"
  | cos => "cos"
  | cosh => "cosh"
  | cumlogsumexp axis reverse => s!"cumlogsumexp {axis} {reverse}"
  | cummax axis reverse => s!"cummax {axis} {reverse}"
  | cummin axis reverse => s!"cummin {axis} {reverse}"
  | cumprod axis reverse => s!"cumprod {axis} {reverse}"
  | cumsum => "cumsum"
  | div => "div"
  | div_int => "div_int"
  | dot_general batch contract lhs rhs => s!"dot_general {contract.length} {batch.length}"
  | dynamic_slice dims => s!"dynamic_slice"
  | dynamic_update_slice dims => s!"dynamic_update_slice"
  | eigvals => "eigvals"
  | eigvalsh => "eigvalsh"
  | eigvecs => "eigvecs"
  | eigvecsh => "eigvecsh"
  | empty => "empty"
  | eq => "eq"
  | erf => "erf"
  | erf_inv => "erf_inv"
  | erfc => "erfc"
  | exp => "exp"
  | exp2 => "exp2"
  | expm1 => "expm1"
  | gather => "gather"
  | iota (n := n) => s!"iota {n}"
  | log => "log"
  | mul => "mul"
  | mod => "mod"
  | neg => "neg"
  | ofNat (σ := ⟨α, s⟩) n => s!"const {α} {s} {n}"
  | scatter => "scatter"
  | sin => "sin"
  | sorted => "sorted"
  | sqrt => "sqrt"
  | sub => "sub"
  | sum n => s!"sum {n}"
  | transpose σ =>
    let σ : List ℕ := List.ofFn fun i => σ i
    s!"transpose {σ}"
  | zeros (σ := σ) => s!"zeros {σ.dtype} {σ.shape}"
  | choice => "where"
  | einsum s indices n => s!"einsum {indices} {n}"
  | flatten => "flatten"
  | unflatten s => s!"unflatten {s}"
  | cast _ => "id"

instance (args : List TensorType) (out : TensorType) : ToString (XlaPrimOp args out) :=
  ⟨XlaPrimOp.toString⟩

instance (exprs : List (List TensorType × List TensorType)) (args outs : List TensorType) :
    ToString (XlaHigherOp exprs args outs) where
  toString op := match op with
  | .repeat => "repeat"
  | .vmap => "vmap"

abbrev XlaOp : SSA.OpType TensorType := SSA.CombineOp (SSA.SimpleOp XlaPrimOp) XlaHigherOp

end Xla
