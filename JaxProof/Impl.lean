import JaxProof.Eval

namespace Jax

abbrev FloatAsReal : TensorType → Type
  | ⟨.float, s⟩ => Tensor ℝ s
  | ⟨.int, s⟩ => Tensor ℤ s

abbrev ArgList := DList FloatAsReal

def FloatAsReal.zero {σ : TensorType} : FloatAsReal σ :=
  match σ with
  | ⟨.float, _⟩
  | ⟨.int, _⟩ => Tensor.zero

instance (σ : TensorType) : Zero (FloatAsReal σ) := ⟨FloatAsReal.zero⟩

def FloatAsReal.get {s : List ℕ} {R : Type} [Zero R]
  (x : Tensor R s) (indices : ArgList (List.replicate s.length ⟨.int, []⟩)) : R :=
  match s with
  | [] => x
  | s₀ :: _ =>
    match s₀ with
    | 0 => 0
    | _ + 1 =>
      let (.cons i₀ i) := indices
      FloatAsReal.get (x (Fin.intCast i₀)) i

def FloatAsReal.gather {α : DType} {s s' : List ℕ}
  (args : ArgList (⟨α, s⟩ :: List.replicate s.length ⟨.int, s'⟩))
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
  ArgList (List.replicate n ⟨.int, s₀ :: s⟩) →
  ArgList (List.replicate n ⟨.int, s⟩) :=
  match n with
  | 0 => id
  | _ + 1 => fun (.cons a₀ as) => .cons (a₀ i₀) <| feed i₀ as

instance (s : Shape) : DecidableEq (ValidIdx s) :=
  inferInstanceAs (DecidableEq (∀ i : Fin s.length, Fin s[i]))

def FloatAsReal.scatter {R : Type} {s : Shape} {n : ℕ} (x : Tensor R s) (y : Tensor R [n])
  (indices : ArgList (List.replicate s.length ⟨.int, [n]⟩)) :
    Tensor R s :=
  let rec get_indices {l m : ℕ} (is : ArgList (List.replicate m ⟨.int, [l]⟩)) :
    Fin m → Fin l → ℤ :=
    match m with
    | 0 => fun r => nomatch r
    | m + 1 => fun r =>
      let (.cons i₀ is) := is
      match r with
      | 0 => i₀
      | .mk (r + 1) hr => get_indices is <| .mk r <| by linarith
  let indices := get_indices indices
  if h : ∀ i : Fin s.length, s[i] ≠ 0 then
    let indices : Fin n → ValidIdx s := fun i r =>
      have : NeZero s[r] := ⟨h r⟩
      Fin.intCast (indices r i)
    let rec update {n : ℕ} (indices : Fin n → ValidIdx s) (x : Tensor R s) (y : Fin n → R) :
      Tensor R s :=
      match n with
      | 0 => x
      | n + 1 =>
        let x := Tensor.of (Function.update x.get (indices 0) (y 0))
        update (Fin.tail indices) x (Fin.tail y)
    update indices x y
  else
    x

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
  | .scatter (α := α) => fun (.cons x (.cons y indices)) =>
    match α with | .int | .float => FloatAsReal.scatter x y indices
  | .iota => fun _ => fun i => (i.val : ℤ)
  | .zeros => 0
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
