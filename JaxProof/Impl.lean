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

def DList.unfold_replicate {α : Type} {γ : α → Type} {n : ℕ} {a : α} :
    DList γ (List.replicate n a) → Fin n → γ a :=
  match n with
  | 0 => fun _ i => nomatch i
  | n + 1 => fun xs i =>
    let (.cons x xs) := xs
    match i with
    | 0 => x
    | .mk (i + 1) h => xs.unfold_replicate <| .mk i <| by omega

def ValidIdx.intCast {s : Shape} (h : ∀ i : Fin s.length, s[i] ≠ 0) (idx : Fin s.length → ℤ) :
    ValidIdx s :=
  fun r =>
    have : NeZero s[r] := ⟨h r⟩
    Fin.intCast (idx r)

def FloatAsReal.scatter {R : Type} {s : Shape} {n : ℕ} (x : Tensor R s) (y : Tensor R [n])
  (indices : ArgList (List.replicate s.length ⟨.int, [n]⟩)) :
    Tensor R s :=
  if h : ∀ i : Fin s.length, s[i] ≠ 0 then
    let indices := (ValidIdx.intCast h) ∘ (Function.swap indices.unfold_replicate)
    Tensor.of fun idx =>
      match Fin.find? (fun i => idx = indices i) with
      | none => x.get idx
      | some i => y i
  else
    x

noncomputable instance : TensorImpl FloatAsReal where
  ofNat i := Int.ofNat i
  toInt x := x
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
  | .div => fun *[x, y] => match out with
    | ⟨.float, _⟩
    | ⟨.int, _⟩ => Tensor.map₂ (· / ·) x y
  | .sum (α := α) n => fun *[x] => match α with
    | .float
    | .int => x.sumN n
  | .sqrt => fun *[x] => x.map Real.sqrt
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
  | .choice (α := α) => fun *[c, x, y]=>
    match α with | .int | .float => Tensor.map₃ (fun c x y => if c != 0 then x else y) c x y
  | .cumsum (σ := σ) => fun *[x] =>
    let ⟨α, s⟩ := σ
    match α with | .int | .float => x.cumsum
  | .ofNat (σ := ⟨α, s⟩) n => fun _ =>
    match α with | .int | .float => Tensor.const n
  | .neg (σ := ⟨α, s⟩) => fun *[x] =>
    match α with | .int | .float => x.map (fun x => - x)
  | .sub (σ := ⟨α, s⟩) => fun *[x, y] =>
    match α with | .int | .float => Tensor.map₂ (fun x y => x - y) x y
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
