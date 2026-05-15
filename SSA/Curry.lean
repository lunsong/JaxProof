import Mathlib.Algebra.Group.Defs

variable {ι : Type}

abbrev Index (m : ι → Type) (γ : List ι) : Type := ∀ i : Fin γ.length, m γ[i]

abbrev Curry (m : ι → Type) (γ : List ι) (α : Type) : Type :=
  match γ with
  | [] => α
  | γ :: γs => m γ → Curry m γs α

variable {m : ι → Type} {α β μ : Type}

def Index.null : Index m [] := fun r => nomatch r

def Index.single {γ : ι} : m γ → Index m [γ] :=
  fun x r => match r with | .mk 0 _ => x

def Index.cons {γ₀ : ι} {γ : List ι} : m γ₀ → Index m γ → Index m (γ₀ :: γ) :=
  fun x₀ x r => match r with
  | .mk 0 h => x₀
  | .mk (r + 1) h => x <| .mk r <| by simpa using h

def Index.select {γ : List ι} (i : List (Fin γ.length)) : Index m γ → Index m (i.map γ.get) :=
  match i with
  | [] => fun x => null
  | i₀ :: i => fun x r =>
    match r with
    | .mk 0 h => x i₀
    | .mk (r + 1) h => select i x <| .mk r <| by simpa using h

def Index.append {γ γ' : List ι} : Index m γ → Index m γ' → Index m (γ ++ γ') :=
  match γ with
  | [] => fun x y => y
  | γ₀ :: γ => fun x y r =>
    match r with
    | .mk 0 h => x <| .mk 0 <| by simp
    | .mk (r + 1) h => append (fun r => x r.succ) y <| .mk r <| by simpa using h

def Index.replicate {i : ι} {n : ℕ} : Index m (List.replicate n i) → Fin n → m i :=
  match n with
  | 0 => fun _ r => nomatch r
  | n + 1 => fun x r =>
    match r with
    | .mk 0 h => x <| .mk 0 <| by simp
    | .mk (r + 1) h =>
      let x' : Index m (List.replicate n i) := fun r => x r.succ
      replicate x' <| .mk r <| by simpa using h

def Curry.get {γ : List ι} (f : Curry m γ α) (i : Index m γ) : α :=
  match γ with
  | [] => f
  | γ :: γs => (f (i ⟨0, by simp⟩)).get fun r => i r.succ

def Curry.of {γ : List ι} (f : Index m γ → α) : Curry m γ α :=
  match γ with
  | [] => f fun r => nomatch r
  | γ :: γs => fun x => of fun a => f fun r =>
    match r with
    | .mk 0 h => x
    | .mk (r + 1) h => a <| .mk r <| by simpa using h

def Index.map {γ : List μ} {f : μ → ι} : Index m (γ.map f) → Index (m ∘ f) γ :=
  match γ with
  | [] => fun _ r => nomatch r
  | _ :: _ => Curry.get <| fun x₀ => Curry.of <| fun x => Index.cons x₀ x.map

def Index.unmap {γ : List μ} {f : μ → ι} : Index (m ∘ f) γ → Index m (γ.map f) :=
  match γ with
  | [] => fun _ r => nomatch r
  | _ :: _ => Curry.get <| fun x₀ => Curry.of <| fun x => Index.cons x₀ x.unmap

theorem Curry.of_get {γ : List ι} (x : Curry m γ α) : of x.get = x := by
  induction γ with
  | nil => rfl
  | cons γ₀ γs ih => simp [of, get, ih, Fin.succ]

theorem Curry.get_of {γ : List ι} (x : Index m γ → α) : (of x).get = x := by
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

def Curry.pure {γ : List ι} (x : α) : Curry m γ α :=
  match γ with
  | [] => x
  | _ :: _ => fun _ => pure x

def Curry.map {γ : List ι} (f : α → β) : Curry m γ α → Curry m γ β :=
  match γ with
  | [] => f
  | _ :: _ => fun a x => (a x).map f

def Curry.map₂ {γ : List ι} (f : α → β → μ) : Curry m γ α → Curry m γ β → Curry m γ μ :=
  match γ with
  | [] => f
  | _ :: _ => fun x y a => map₂ f (x a) (y a)

def Curry.bind {γ : List ι} (x : Curry m γ α) (f : α → Curry m γ β) : Curry m γ β :=
  match γ with
  | [] => f x
  | _ :: _ => fun a => bind (x a) fun b => f b a

def Curry.arg {γ : List ι} (i : Fin γ.length) : Curry m γ (m γ[i]) :=
  match γ with
  | γ₀ :: γs =>
    match i with
    | .mk 0 _ => fun x => pure x
    | .mk (i + 1) hi => fun _ => arg <| .mk i <| by simpa using hi

instance Curry.instMonad (γ : List ι) : Monad (Curry m γ) where
  pure := Curry.pure
  bind := Curry.bind

instance Curry.instZero (γ : List ι) [Zero α] : Zero (Curry m γ α) where
  zero := pure 0

instance Curry.instHAdd (γ : List ι) [HAdd α β μ] :
    HAdd (Curry m γ α) (Curry m γ β) (Curry m γ μ) where
  hAdd x y := do return (← x) + (← y)

instance Curry.instAdd (γ : List ι) [Add α] : Add (Curry m γ α) where
  add x y := do return (← x) + (← y)

instance Curry.instAddCommMonoid (γ : List ι) [AddCommMonoid α] :
    AddCommMonoid (Curry m γ α) where
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

def Curry.curry {γ γ' : List ι} : Curry m (γ ++ γ') α → Curry m γ (Curry m γ' α) := 
  match γ with
  | [] => id
  | _ :: _ => fun x a => (x a).curry

def Curry.uncurry {γ γ' : List ι} : Curry m γ (Curry m γ' α) → Curry m (γ ++ γ') α :=
  match γ with
  | [] => id
  | _ :: _ => fun x a => (x a).uncurry

def Curry.transpose {γ γ' : List ι} : Curry m (γ ++ γ') α → Curry m (γ' ++ γ) α :=
  fun x => uncurry <| of <| fun i => of <| fun j => (x.curry.get j).get i

def Curry.transposeFirst {γ₀ : ι} {γ : List ι} : Curry m (γ₀ :: γ) α → Curry m γ (m γ₀ → α) :=
  fun x => curry (γ' := [γ₀]) <| transpose x

