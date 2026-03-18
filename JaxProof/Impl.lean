import JaxProof.Eval

namespace Jax

abbrev FloatAsReal : TensorType → Type
  | ⟨.float, s⟩ => Tensor ℝ s
  | ⟨.int, s⟩ => Tensor ℤ s

def FloatAsReal.zero {σ : TensorType} : FloatAsReal σ :=
  match σ with
  | ⟨.float, _⟩
  | ⟨.int, _⟩ => Tensor.zero

instance (σ : TensorType) : Zero (FloatAsReal σ) := ⟨FloatAsReal.zero⟩

#check Tensor.transpose

def FloatAsReal.get {s : List ℕ} {R : Type} [Zero R]
  (x : Tensor R s) (indices : DList FloatAsReal (List.replicate s.length ⟨.int, []⟩)) : R :=
  match s with
  | [] => x
  | s₀ :: _ =>
    match s₀ with
    | 0 => 0
    | _ + 1 =>
      let (.cons i₀ i) := indices
      FloatAsReal.get (x (Fin.intCast i₀)) i

def FloatAsReal.gather {α : DType} {s s' : List ℕ}
  (args : DList FloatAsReal (⟨α, s⟩ :: List.replicate s.length ⟨.int, s'⟩))
  : FloatAsReal ⟨α, s'⟩ :=
  let (.cons x indices) := args
  match α with
  | .float | .int =>
    match s' with
    | [] =>
      FloatAsReal.get x indices
    | _ :: _ => fun i₀ => 
      let indices := feed i₀ indices
      FloatAsReal.gather (.cons x indices)
where feed {n : ℕ} {s₀ : ℕ} {s : List ℕ} (i₀ : Fin s₀) :
  DList FloatAsReal (List.replicate n ⟨.int, s₀ :: s⟩) →
  DList FloatAsReal (List.replicate n ⟨.int, s⟩) :=
  match n with
  | 0 => id
  | _ + 1 => fun (.cons a₀ as) => .cons (a₀ i₀) <| feed i₀ as


noncomputable instance : TensorImpl FloatAsReal where
  ofNat i := Int.ofNat i
  impl {args} {out} op := match op with
  | .abs => fun *[x] => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => x.map abs
  | .acos => fun *[x] => x.map Real.arccos
  -- `Real.Arcosh` isn't available in this lean version
  | .acosh => fun *[x] => x.map fun x => Real.log (x + Real.sqrt (x^2 + 1))
  | .add => fun *[x, y] => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.add x y
  | .mul => fun *[x, y] => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.map₂ (· * ·) x y
  | .sum (α := α) n => fun *[x] => match α with
    | .float
    | .int => x.sumN n
  | .transpose (α := α) σ => fun *[x] => match α with
    | .float | .int => x.transpose σ
  | .broadcast (α := α) s => fun *[x] => match α with
    | .float | .int => x.broadcast
  | .dot_general (α := α) batch contract lhs rhs => fun *[x,y] =>
    match α with
    | .float
    | .int =>
      let x : Tensor _ (contract ++ batch ++ lhs ++ rhs) := Tensor.uncurry' (x.map Tensor.const)
      let y : Tensor _ (contract ++ batch ++ (lhs ++ rhs)) :=
        Tensor.uncurry' <| (y.curry'.map Tensor.const).map Tensor.uncurry'
      let y : Tensor _ (contract ++ batch ++ lhs ++ rhs) := y.cast <| by simp
      let z := (Tensor.map₂ (· * ·) x y).sumN (contract.length)
      z.cast <| by simp
  | .gather (α := α) (s := s) => FloatAsReal.gather
  | .sorted (α := α) => fun *[x] => match α with
    | .int =>
      Tensor.uncurry' <|
        x.curry'.map fun (x : Fin _ → ℤ) i =>
          ((List.ofFn x).mergeSort).get <| i.cast <| by simp
    | .float =>
      Tensor.uncurry' <|
        x.curry'.map fun (x : Fin _ → ℝ) i =>
          ((List.ofFn x).mergeSort).get <| i.cast <| by simp
    --match α with
    --| .float =>
    --  match s with
    --  | [] => fun *[x] => Tensor.const x
    --  | s :: s' => 
--  | .dot_general (α := α) batch contract lhs rhs => fun ⟨x, y, _⟩ =>
--    match α with
--    | .float =>
--      let real_indices (x : List ℕ) : List (ℕ × Bool) := x.map (Prod.mk · true) 
--      let virt_indices (x : List ℕ) : List (ℕ × Bool) := x.map (Prod.mk · false) 
--      have preBroadcast_real (x : List ℕ) : Tensor.preBroadcast (real_indices x) = x := by
--        induction x with
--        | nil => rfl
--        | cons x xs ih =>
--          unfold real_indices
--          simp [Tensor.preBroadcast]
--          exact ih
--      have preBroadcast_virt (x : List ℕ) : Tensor.preBroadcast (virt_indices x) = [] := by
--        induction x with
--        | nil => rfl
--        | cons x xs ih =>
--          unfold virt_indices
--          simp [Tensor.preBroadcast]
--      let lhs_broadcast := real_indices (contract ++ batch ++ lhs) ++ virt_indices rhs
--      have preBroadcast_lhs : Tensor.preBroadcast lhs_broadcast = contract ++ batch ++ lhs := by
--        unfold lhs_broadcast
--        rw [Tensor.preBroadcast_append, preBroadcast_real, preBroadcast_virt, List.append_nil]
--      let x' := (x.cast preBroadcast_lhs.symm).broadcast lhs_broadcast
--      by
--
--      sorry
  | _ => 0


end Jax
