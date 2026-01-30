import JaxProof.Tensor

namespace Jax

inductive DataType where
  | float : DataType
  | int : DataType

/-
class TensorImpl (α : List ℕ → DataType → Type) where
  einsum {σ : DataType} (s : List ℕ)
    (xs : List ((i : List (Fin s.length)) × α (i.map s.get) σ)) (nsum : ℕ) : α (s.drop nsum) σ
  exp {s : List ℕ} : α s .float → α s .float
  div {s : List ℕ} : α s .float → α s .float → α s .float
  broadcast {σ : DataType} (s : List (ℕ × Bool)) :
    α (Tensor.preBroadcast s) σ → α (s.map Prod.fst) σ
-/

class TensorImpl (α : List ℕ → Type) where
  einsum (s : List ℕ)
    (xs : List ((i : List (Fin s.length)) × α (i.map s.get))) (nsum : ℕ) : α (s.drop nsum)
  exp {s : List ℕ} : α s → α s
  div {s : List ℕ} : α s → α s → α s
  broadcast (s : List (ℕ × Bool)) :
    α (Tensor.preBroadcast s) → α (s.map Prod.fst)

def FloatAsReal : DataType → Type
  | .float => ℝ
  | .int => ℤ

instance (σ : DataType) : AddCommMonoid (FloatAsReal σ) :=
  match σ with
  | .float => inferInstanceAs (AddCommMonoid ℝ)
  | .int   => inferInstanceAs (AddCommMonoid ℤ)

instance (σ : DataType) : Mul (FloatAsReal σ) :=
  match σ with
  | .float => inferInstanceAs (Mul ℝ)
  | .int   => inferInstanceAs (Mul ℤ)

instance (σ : DataType) : One (FloatAsReal σ) :=
  match σ with
  | .float => inferInstanceAs (One ℝ)
  | .int   => inferInstanceAs (One ℤ)

def NativeTensor : List ℕ → Type := fun s ↦ Tensor ℝ s

@[simps]
noncomputable instance NativeImpl : TensorImpl NativeTensor where
  einsum := Tensor.einsum
  broadcast := Tensor.broadcast
  exp := Tensor.map Real.exp
  div := Tensor.map₂ fun (x y : ℝ) ↦ x / y

mutual

inductive Tracer : List ℕ → Type where
  | arg {s : List ℕ} : ℕ → Tracer s
  | div {s : List ℕ} : Tracer s → Tracer s → Tracer s
  | broadcast (s : List (ℕ × Bool)) :
    Tracer (Tensor.preBroadcast s) → Tracer (s.map Prod.fst)
  | einsum (s : List ℕ) (xs : Tracer.einsumArg s) (nsum : ℕ) : Tracer (s.drop nsum)
  | exp {s : List ℕ} : Tracer s → Tracer s

inductive Tracer.einsumArg : List ℕ → Type where
  | nil {s : List ℕ} : Tracer.einsumArg s
  | cons {s : List ℕ} (i : List (Fin s.length)) :
    Tracer (i.map s.get) → Tracer.einsumArg s → Tracer.einsumArg s

end

mutual

def Tracer.toString {s : List ℕ} : Tracer s → String
  | arg n => s!"x{n}"
  | div x y => s!"{x.toString} / {y.toString}"
  | exp x => s!"exp({x.toString})"
  | broadcast s x =>
    let reshape : List ℕ := s.map fun ⟨n, is_broadcast⟩ ↦ if is_broadcast then 1 else n
    s!"broadcast({x.toString}.reshape({reshape}), {s.map Prod.fst})"
  | einsum s xs nsum => s!"einsum({s}, {xs.toString}, {nsum})"

def Tracer.einsumArg.toString {s : List ℕ} : Tracer.einsumArg s → String
  | .nil => ""
  | .cons _ x xs => s!"{x.toString},{xs.toString}"

end

def Tracer.einsumArg.of {s : List ℕ} :
    List ((i : List (Fin s.length)) × Tracer (i.map s.get)) → einsumArg s
  | [] => nil
  | ⟨i, x⟩ :: xs => cons i x (of xs)

instance TracerImpl : TensorImpl Tracer where
  einsum s xs := Tracer.einsum s (Tracer.einsumArg.of xs)
  broadcast := Tracer.broadcast
  exp := Tracer.exp
  div := Tracer.div

def Softmax {α : List ℕ → Type} [TensorImpl α] {n₁ n₂ : ℕ} (x : α [n₁, n₂]) :
    α [n₁, n₂] :=
  let x₁ := TensorImpl.exp x
  let normalizer : α [n₁] := TensorImpl.einsum [n₂, n₁] [⟨[#1,#0], x₁⟩] 1
  let denom : α [n₁, n₂] := TensorImpl.broadcast [(n₁, false), (n₂, true)] normalizer
  TensorImpl.div x₁ denom

example (n₁ n₂ : ℕ) (x : Tensor ℝ [n₁, n₂]) (i : Fin n₁) (j : Fin n₂) :
    Softmax (α := NativeTensor) x i j = Real.exp (x i j) / ∑ k, Real.exp (x i k) := by
  simp [Softmax, Tensor.map₂, Tensor.map, Tensor.broadcast, Tensor.einsum, Tensor.einprod]
  apply congrArg
  conv_lhs =>
    change (∑ j, fun i ↦ Real.exp (x i j)) i
  simp

def Softmax_traced (n₁ n₂ : ℕ) : Tracer [n₁, n₂] :=
  Softmax (Tracer.arg 0 : Tracer [n₁, n₂])

#eval IO.println (Softmax_traced 10 20).toString

end Jax
