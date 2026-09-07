import Mathlib.Algebra.Group.Defs
import Soir.Meta

namespace Soir

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

@[reduce_soir]
theorem Index.single_zero {ι : Type} {m : ι → Type} {i : ι} {x : m i} : Index.single x 0 = x := rfl

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

@[reduce_soir]
def Index.map {γ : List μ} {f : μ → ι} : Index m (γ.map f) → Index (m ∘ f) γ :=
  match γ with
  | [] => fun _ r => nomatch r
  | _ :: _ => Curry.get <| fun x₀ => Curry.of <| fun x => Index.cons x₀ x.map

@[reduce_soir]
def Index.unmap {γ : List μ} {f : μ → ι} : Index (m ∘ f) γ → Index m (γ.map f) :=
  match γ with
  | [] => fun _ r => nomatch r
  | _ :: _ => Curry.get <| fun x₀ => Curry.of <| fun x => Index.cons x₀ x.unmap

@[reduce_soir]
theorem Curry.of_get {γ : List ι} (x : Curry m γ α) : of x.get = x := by
  induction γ with
  | nil => rfl
  | cons γ₀ γs ih => simp [of, get, ih, Fin.succ]

@[reduce_soir]
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

@[reduce_soir]
def Curry.pure {γ : List ι} (x : α) : Curry m γ α :=
  match γ with
  | [] => x
  | _ :: _ => fun _ => pure x

@[reduce_soir]
def Curry.map {γ : List ι} (f : α → β) : Curry m γ α → Curry m γ β :=
  match γ with
  | [] => f
  | _ :: _ => fun a x => (a x).map f

@[reduce_soir]
def Curry.map₂ {γ : List ι} (f : α → β → μ) : Curry m γ α → Curry m γ β → Curry m γ μ :=
  match γ with
  | [] => f
  | _ :: _ => fun x y a => map₂ f (x a) (y a)

@[reduce_soir]
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

/-!
### Computation rules for `Curry`/`Index` at concrete positions

The definitions of `Curry.get`, `Curry.of`, `Index.append` and `Curry.arg` are
matches on the index list and on `Fin` values. Unfolding them with `simp`
leaves dependent matches whose scrutinees are `OfNat` numerals (the simp
normal form for `Fin` literals, see `Fin.zero_eta`/`Fin.mk_one`); those matches
reduce neither by `simp` nor by the kernel. The lemmas below instead compute
applications at concrete positions (`0`, `Fin.succ r`) directly, so evaluation
never exposes the matches.
-/

@[reduce_soir]
theorem Curry.get_zero (f : Curry m [] α) (i : Index m []) : f.get i = f := rfl

@[reduce_soir]
theorem Curry.get_one {γ₀ : ι} (f : Curry m [γ₀] α) (i : Index m [γ₀]) :
    f.get i = f (i 0) := rfl

@[reduce_soir]
theorem Curry.get_two {γ₀ γ₁ : ι} (f : Curry m [γ₀, γ₁] α) (i : Index m [γ₀, γ₁]) :
    f.get i = f (i 0) (i 1) := rfl

@[reduce_soir]
theorem Curry.get_three {γ₀ γ₁ γ₂ : ι} (f : Curry m [γ₀, γ₁, γ₂] α)
    (i : Index m [γ₀, γ₁, γ₂]) :
    f.get i = f (i 0) (i 1) (i 2) := rfl

@[reduce_soir]
theorem Curry.get_pure {γ : List ι} (x : α) (i : Index m γ) : (Curry.pure x).get i = x := by
  induction γ with
  | nil => rfl
  | cons γ₀ γs ih => exact ih fun r ↦ i r.succ

@[reduce_soir]
theorem Curry.get_map {γ : List ι} (f : α → β) (g : Curry m γ α) (i : Index m γ) :
    (g.map f).get i = f (g.get i) := by
  induction γ with
  | nil => rfl
  | cons γ₀ γs ih => exact ih (g (i ⟨0, by simp⟩)) fun r ↦ i r.succ

@[reduce_soir]
theorem Index.cons_zero {γ₀ : ι} {γ : List ι} (x₀ : m γ₀) (x : Index m γ) :
    Index.cons x₀ x 0 = x₀ :=
  show Index.cons x₀ x ⟨0, by simp⟩ = x₀ from rfl

@[reduce_soir]
theorem Index.cons_one {γ₀ γ₁ : ι} {γ : List ι} (x₀ : m γ₀) (x : Index m (γ₁ :: γ)) :
    Index.cons x₀ x 1 = x 0 :=
  show Index.cons x₀ x (Fin.succ 0) = x 0 from rfl

@[reduce_soir]
theorem Index.cons_two {γ₀ γ₁ γ₂ : ι} {γ : List ι} (x₀ : m γ₀)
    (x : Index m (γ₁ :: γ₂ :: γ)) : Index.cons x₀ x 2 = x 1 :=
  show Index.cons x₀ x (Fin.succ 1) = x 1 from rfl

@[reduce_soir]
theorem Index.cons_succ {γ₀ : ι} {γ : List ι} (x₀ : m γ₀) (x : Index m γ) (r : Fin γ.length) :
    Index.cons x₀ x r.succ = x r := rfl

@[reduce_soir]
theorem Index.append_null {γ : List ι} (y : Index m γ) : Index.append Index.null y = y := rfl

@[reduce_soir]
theorem Index.append_single {γ₀ : ι} {γ : List ι} (x₀ : m γ₀) (y : Index m γ) :
    Index.append (Index.single x₀) y = Index.cons x₀ y := by
  funext ⟨r, hr⟩
  match r, hr with
  | 0, h => rfl
  | r + 1, h => rfl

@[reduce_soir]
theorem Index.append_cons {γ₀ : ι} {γ γ' : List ι} (x₀ : m γ₀) (x : Index m γ)
    (y : Index m γ') :
    Index.append (Index.cons x₀ x) y = Index.cons x₀ (Index.append x y) := by
  funext ⟨r, hr⟩
  match r, hr with
  | 0, h => rfl
  | r + 1, h => rfl

@[reduce_soir]
theorem Curry.of_apply_zero (f : Index m [] → α) :
    (Curry.of f : Curry m [] α) = f Index.null := rfl

@[reduce_soir]
theorem Curry.of_apply_one {γ₀ : ι} (f : Index m [γ₀] → α) (x₀ : m γ₀) :
    (Curry.of f : Curry m [γ₀] α) x₀ = f (Index.single x₀) := by
  rw [show f (Index.single x₀) = (Curry.of f).get (Index.single x₀) from by rw [Curry.get_of],
    Curry.get_one, Index.single_zero]

@[reduce_soir]
theorem Curry.of_apply_two {γ₀ γ₁ : ι} (f : Index m [γ₀, γ₁] → α) (x₀ : m γ₀) (x₁ : m γ₁) :
    (Curry.of f : Curry m [γ₀, γ₁] α) x₀ x₁ = f (Index.cons x₀ (Index.single x₁)) := by
  rw [show f (Index.cons x₀ (Index.single x₁)) =
      (Curry.of f).get (Index.cons x₀ (Index.single x₁)) from by rw [Curry.get_of],
    Curry.get_two, Index.cons_zero, Index.cons_one, Index.single_zero]

@[reduce_soir]
theorem Curry.arg_zero {γ₀ : ι} {γ : List ι} :
    Curry.arg (m := m) (0 : Fin (γ₀ :: γ).length) = fun x ↦ Curry.pure x :=
  show Curry.arg (m := m) (γ := γ₀ :: γ) ⟨0, by simp⟩ = fun x ↦ Curry.pure x from rfl

@[reduce_soir]
theorem Curry.arg_one {γ₀ γ₁ : ι} {γ : List ι} :
    Curry.arg (m := m) (1 : Fin (γ₀ :: γ₁ :: γ).length) = fun _ x ↦ Curry.pure x :=
  show Curry.arg (m := m) (γ := γ₀ :: γ₁ :: γ) (Fin.succ 0) = fun _ x ↦ Curry.pure x from rfl

@[reduce_soir]
theorem Curry.arg_two {γ₀ γ₁ γ₂ : ι} {γ : List ι} :
    Curry.arg (m := m) (2 : Fin (γ₀ :: γ₁ :: γ₂ :: γ).length) = fun _ _ x ↦ Curry.pure x :=
  show Curry.arg (m := m) (γ := γ₀ :: γ₁ :: γ₂ :: γ) (Fin.succ 1)
    = fun _ _ x ↦ Curry.pure x from rfl

@[reduce_soir]
theorem Curry.arg_succ {γ₀ : ι} {γ : List ι} (i : Fin γ.length) :
    Curry.arg (m := m) (γ := γ₀ :: γ) i.succ = fun _ ↦ Curry.arg i := rfl

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
    | nil => exact zero_add x
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)
  add_zero x := by
    induction γ with
    | nil => exact add_zero x
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)
  add_comm x y := by
    induction γ with
    | nil => exact add_comm x y
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i) (y i)
  add_assoc x y z := by
    induction γ with
    | nil => exact add_assoc x y z
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i) (y i) (z i)
  nsmul n x := do return n • (← x)
  nsmul_zero x := by
    induction γ with
    | nil => exact AddMonoid.nsmul_zero x
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)
  nsmul_succ n x := by
    induction γ with
    | nil => exact AddMonoid.nsmul_succ n x
    | cons γ₀ γs ih =>
      refine funext fun i => ?_
      exact ih (x i)

@[reduce_soir]
def Curry.curry {γ γ' : List ι} : Curry m (γ ++ γ') α → Curry m γ (Curry m γ' α) := 
  match γ with
  | [] => id
  | _ :: _ => fun x a => (x a).curry

@[reduce_soir]
def Curry.uncurry {γ γ' : List ι} : Curry m γ (Curry m γ' α) → Curry m (γ ++ γ') α :=
  match γ with
  | [] => id
  | _ :: _ => fun x a => (x a).uncurry

def Curry.transpose {γ γ' : List ι} : Curry m (γ ++ γ') α → Curry m (γ' ++ γ) α :=
  fun x => uncurry <| of <| fun i => of <| fun j => (x.curry.get j).get i

def Curry.transposeFirst {γ₀ : ι} {γ : List ι} : Curry m (γ₀ :: γ) α → Curry m γ (m γ₀ → α) :=
  fun x => curry (γ' := [γ₀]) <| transpose x

end Soir
