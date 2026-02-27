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

structure TensorInfo where
  dtype : DataType
  shape : List ℕ

class TensorProgram (impl : outParam (TensorInfo → Type))
  (program : List TensorInfo → Type → Type) where
  monadic {args : List TensorInfo} : Monad (program args)
  arg {args : List TensorInfo} (i : Fin args.length) : program args (impl args[i])
  sin {args : List TensorInfo} {s : List ℕ} : impl ⟨.float, s⟩ → program args (impl ⟨.float, s⟩)
  add {args : List TensorInfo} {σ : TensorInfo} :
    impl σ → impl σ → program args (impl σ)

instance (impl : TensorInfo → Type) (program : List TensorInfo → Type → Type)
  [TensorProgram impl program] (args : List TensorInfo) : Monad (program args) :=
  TensorProgram.monadic (args := args)

inductive TensorOp (info : TensorInfo) : Type where
  | arg : ℕ → TensorOp info
  | var : ℕ → TensorOp info

inductive TensorCommand where
  | sin {s : List ℕ} : TensorOp ⟨.float, s⟩ → TensorCommand
  | add {σ : TensorInfo} : TensorOp σ → TensorOp σ → TensorCommand

set_option linter.unusedVariables false in
def TensorStackProgram (args : List TensorInfo) := StateM (List TensorCommand)

instance : TensorProgram TensorOp TensorStackProgram where
  monadic := StateT.instMonad
  arg i := fun l ↦ ⟨.arg i.val, l⟩
  sin x := fun l ↦ ⟨.var l.length, l.concat (.sin x)⟩
  add x y := fun l ↦ ⟨.var l.length, l.concat (.add x y)⟩

/-
inductive TensorCommand : TensorInfo → Type
  -- unops
  | abs {info : TensorInfo} : TensorOp info→ TensorCommand info
  | neg {info : TensorInfo} : TensorOp info→ TensorCommand info
  | sqrt {info : TensorInfo} : TensorOp info→ TensorCommand info
  | cbrt {info : TensorInfo} : TensorOp info→ TensorCommand info
  | exp {info : TensorInfo} : TensorOp info→ TensorCommand info
  | log {info : TensorInfo} : TensorOp info→ TensorCommand info
  | sin {info : TensorInfo} : TensorOp info→ TensorCommand info
  | cos {info : TensorInfo} : TensorOp info→ TensorCommand info
  -- binops
  | add {info : TensorInfo} : TensorOp info→ TensorOp info→ TensorCommand info
  | mul {info : TensorInfo} : TensorOp info→ TensorOp info→ TensorCommand info
  | div {info : TensorInfo} : TensorOp info→ TensorOp info→ TensorCommand info
  | pow {info : TensorInfo} : TensorOp info→ TensorOp info→ TensorCommand info

structure TensorStorage where
  info : TensorInfo
  commabd : TensorCommand info

set_option linter.unusedVariables false in
def TensorProgram (args : List TensorInfo) := StateM (List TensorStorage)

instance (args : List TensorInfo) : Monad (TensorProgram args) :=
  inferInstanceAs (Monad (StateM (List TensorStorage)))

def stackPush {args : List TensorInfo} {info : TensorInfo} (x : TensorCommand info) :
    TensorProgram args (TensorOp info) :=
  fun l ↦ ⟨.var l.length, l.concat ⟨info, x⟩⟩

def getArg {args : List TensorInfo} (i : Fin args.length) : TensorProgram args (TensorOp args[i]) :=
  fun l ↦ ⟨.arg i.val, l⟩

def sin {args : List TensorInfo} {info : TensorInfo} (x : TensorOp info) :
    TensorProgram args (TensorOp info) :=
  stackPush (.sin x)

def add {args : List TensorInfo} {info : TensorInfo} (x y : TensorOp info) :
    TensorProgram args (TensorOp info) :=
  stackPush (.add x y)

def simple_program (impl : TensorInfo → Type) (program : List TensorInfo → Type → Type)
  [TensorProgram impl program] :
    program [⟨.float, [2,3]⟩, ⟨.float, [2,3]⟩] (impl ⟨.float, [2,3]⟩) := do
  let x ← TensorProgram.arg 0
  let y ← TensorProgram.arg 1
  let z ← TensorProgram.add x y
  TensorProgram.add z x

instance : ToString TensorOp where
  toString op := match op with
  | .arg n => s!"arg {n}"
  | .var n => s!"var {n}"

#eval IO.println (⟨.arg 0, .arg 1⟩ : TensorOp × TensorOp)

def stackPush {α : Type} (x : α) : StateM (List α) ℕ := fun l ↦ ⟨l.length, l.concat x⟩

instance : TensorProgram (fun _ ↦ TensorOp) (fun _ ↦ StateM (List (TensorOp × TensorOp))) where
  monadic := inferInstance
  arg i := pure (.arg i.val)
  add x y := do
    let i ← stackPush ⟨x, y⟩
    return .var i

#eval simple_program (fun _ ↦ TensorOp) (fun _ ↦ StateM (List (TensorOp × TensorOp))) []
-/

end Jax
