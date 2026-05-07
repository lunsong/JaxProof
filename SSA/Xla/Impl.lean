import SSA.Core
import SSA.Xla.Op

namespace Xla

open SSA

abbrev DirectImpl : TensorType → Type
  | ⟨.float, s⟩ => Tensor ℝ s
  | ⟨.int, s⟩ => Tensor ℤ s

def DirectImpl.gather {α : DType} {s s' : Shape} :
    Curry DirectImpl (⟨α, s⟩ :: List.replicate s.length ⟨.int, s'⟩) (DirectImpl ⟨α, s'⟩) :=
  match α with
  | .int =>
    match s with
    | [] => Curry.pure
    | s₀ :: _ => fun x i => Curry.of <| fun r => Curry.of <| fun r' =>
        if hs : s₀ = 0 then 0 else
          have : NeZero s₀ := ⟨hs⟩
          let x  := x <| Fin.intCast (i.get r');
          Curry.get ((gather (α := .int) x).get r) r'
  | .float =>
    match s with
    | [] => Curry.pure
    | s₀ :: _ => fun x i => Curry.of <| fun r => Curry.of <| fun r' =>
        if hs : s₀ = 0 then 0 else
          have : NeZero s₀ := ⟨hs⟩
          let x := x <| Fin.intCast (i.get r');
          Curry.get ((gather (α := .float) x).get r) r'

def DirectImpl.scatter {α : DType} {s : Shape} {n : ℕ}
  (x : DirectImpl ⟨α, s⟩) (y : DirectImpl ⟨α, [n]⟩) :
    Curry DirectImpl (List.replicate s.length ⟨.int, [n]⟩) (DirectImpl ⟨α, s⟩) :=
  match α with
  | .int
  | .float =>
    if hs : ∀ i : Fin s.length, s[i] ≠ 0 then
      have (i : Fin s.length) : NeZero s[i] := ⟨hs i⟩
      Curry.of <| fun i =>
      let i : Fin n → Index Fin s := fun n r => Fin.intCast <| i.replicate r n
      Curry.of fun r => match Fin.find? fun n => i n = r with
      | some n => y n
      | none => x.get r
   else
      Curry.pure x

def DirectImpl.zero {args : List TensorType} {out : TensorType} :
    Curry DirectImpl args (DirectImpl out) :=
  Curry.pure <|
    match out with
    | ⟨.int, _⟩
    | ⟨.float, _⟩ => Curry.pure 0

noncomputable instance : SimpleImpl XlaPrimOp DirectImpl where
  bind op := match op with
  |.abs (σ := ⟨α, n⟩) => fun x => match α with | .float | .int => x.map abs
  -- `Real.Arcosh` isn't available in this lean version
  | .acosh => fun x => x.map fun x => Real.log (x + Real.sqrt (x^2 + 1))
  | .add (σ := σ) => fun x y => match σ with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.map₂ (· + ·) x y
  | .mul (σ := σ) => fun x y => match σ with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.map₂ (· * ·) x y
  | .div (σ := σ) => fun x y => match σ with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.map₂ (· / ·) x y
  | .sum (α := α) n => fun x => match α with
    | .float
    | .int => x.sumN n
  | .sqrt => fun x => x.map Real.sqrt
  | .transpose (α := α) σ => fun x => match α with
    | .float | .int => x.transpose σ
  | .broadcast (α := α) s => fun x => match α with
    | .float | .int => x.broadcast
  | .dot_general (α := α) batch contract lhs rhs => fun x y =>
    match α with
    | .float
    | .int =>
      let x : Tensor _ (contract ++ batch ++ lhs ++ rhs) := Tensor.uncurry' (x.map Curry.pure)
      let y : Tensor _ (contract ++ batch ++ (lhs ++ rhs)) :=
        Tensor.uncurry' <| (y.curry'.map Curry.pure).map Tensor.uncurry'
      let y : Tensor _ (contract ++ batch ++ lhs ++ rhs) := y.cast <| by simp
      let z := (Tensor.map₂ (· * ·) x y).sumN (contract.length)
      z.cast <| by simp
  | .gather  => DirectImpl.gather
  | .sorted (α := α) => fun x => match α with
    | .int =>
      Tensor.uncurry' <|
        x.curry'.map fun (x : Fin _ → ℤ) i =>
          ((List.ofFn x).mergeSort).get <| i.cast <| by simp
    | .float =>
      Tensor.uncurry' <|
        x.curry'.map fun (x : Fin _ → ℝ) i =>
          ((List.ofFn x).mergeSort).get <| i.cast <| by simp
  | .scatter => DirectImpl.scatter
  | .iota => fun i => i
  | .zeros (σ := σ) => match σ with | ⟨.int, _⟩ | ⟨.float, _⟩ => Curry.pure 0
  | .choice (α := α) => fun c x y=>
    match α with | .int | .float => Tensor.map₃ (fun c x y => if c != 0 then x else y) c x y
  | .cumsum (σ := σ) => fun x =>
    let ⟨α, s⟩ := σ
    match α with | .int | .float => x.cumsum
  | .ofNat (σ := ⟨α, s⟩) n => match α with | .int | .float => Curry.pure (m := Fin) n
  | .neg (σ := ⟨α, s⟩) => fun x =>
    match α with | .int | .float => x.map (fun x => - x)
  | .sub (σ := ⟨α, s⟩) => fun x y =>
    match α with | .int | .float => Tensor.map₂ (fun x y => x - y) x y
  | _ => DirectImpl.zero

instance : Impl XlaRepeatOp DirectImpl where
  bind op := match op with
  | .repeat => fun fn n => Curry.uncurry <| Curry.of <| fun x => Curry.of <| fun aux =>
    let n : ℕ := n.natAbs
    let fn := (fn.transpose.curry.get aux).get
    Nat.repeat fn n x

end Xla
