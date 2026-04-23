import Mathlib.Algebra.Group.Defs

def Index (γ : List Type) : Type := ∀ i : Fin γ.length, γ[i]

def Curry (γ : List Type) (α : Type) : Type :=
  match γ with
  | [] => α
  | γ :: γs => γ → Curry γs α

def Curry.get {γ : List Type} {α : Type} (f : Curry γ α) (i : Index γ) : α :=
  match γ with
  | [] => f
  | γ :: γs => (f (i ⟨0, by simp⟩)).get fun r => i r.succ

def Curry.of {γ : List Type} {α : Type} (f : Index γ → α) : Curry γ α :=
  match γ with
  | [] => f fun r => nomatch r
  | γ :: γs => fun x => of fun a => f fun r =>
    match r with
    | .mk 0 h => x
    | .mk (r + 1) h => a <| .mk r <| by simpa using h

theorem Curry.of_get {γ : List Type} {α : Type} (x : Curry γ α) : of x.get = x := by
  induction γ with
  | nil => rfl
  | cons γ₀ γs ih => simp [of, get, ih, Fin.succ]

theorem Curry.get_of {γ : List Type} {α : Type} (x : Index γ → α) : (of x).get = x := by
  ext i
  induction γ with
  | nil =>
    simp only [get, of, List.length_nil, Fin.getElem_fin]
    congr
    refine funext fun r => ?_
    nomatch r
  | cons γ₀ γs ih =>
    simp only [get, of, List.length_cons, Fin.getElem_fin, Fin.zero_eta, ih, Fin.succ_mk]
    congr
    refine funext fun r => ?_
    match r with | 0 | .mk (r + 1) h => rfl

def Curry.pure {γ : List Type} {α : Type} (x : α) : Curry γ α :=
  match γ with
  | [] => x
  | _ :: _ => fun _ => pure x

def Curry.bind {γ : List Type} {α β : Type} (x : Curry γ α) (f : α → Curry γ β) : Curry γ β :=
  match γ with
  | [] => f x
  | _ :: _ => fun a => bind (x a) fun b => f b a

instance Curry.instMonad (γ : List Type) : Monad (Curry γ) where
  pure := Curry.pure
  bind := Curry.bind

instance Curry.instZero (γ : List Type) (α : Type) [Zero α] : Zero (Curry γ α) where
  zero := pure 0

instance Curry.instHAdd (γ : List Type) (α β μ : Type) [HAdd α β μ] :
    HAdd (Curry γ α) (Curry γ β) (Curry γ μ) where
  hAdd x y := do return (← x) + (← y)

instance Curry.instAdd (γ : List Type) (α : Type) [Add α] : Add (Curry γ α) where
  add x y := do return (← x) + (← y)

instance Curry.instAddCommMonoid (γ : List Type) (α : Type) [AddCommMonoid α] :
    AddCommMonoid (Curry γ α) where
  zero_add x := by
    induction γ with
    | nil => simp
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)
  add_zero x := by
    induction γ with
    | nil => simp
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)
  add_comm x y := by
    induction γ with
    | nil => simp [add_comm]
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i) (y i)
  add_assoc x y z := by
    induction γ with
    | nil => simp [add_assoc]
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i) (y i) (z i)
  nsmul n x := do return n • (← x)
  nsmul_zero x := by
    induction γ with
    | nil => simp; rfl
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)
  nsmul_succ n x := by
    induction γ with
    | nil => simp [Pure.pure, Bind.bind, bind, pure, succ_nsmul]
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)

