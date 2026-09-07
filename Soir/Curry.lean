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

/-!
### A simproc for eliminating `Index` constructors at literal positions

Applications of an `Index` built from `Index.single`/`Index.cons`/`Index.append`/
`Index.map`/`Index.unmap` at a concrete position (`a 0`, `a 1`, ...) cannot be
simplified by the rewrite lemmas above (`Index.cons_zero`, `Index.append_single`,
...): `simp` matches at `implicit` transparency, and the *types* of these terms
involve `List.append`/`List.length`/`List.getElem` computations that only reduce at
default transparency, so the matches are never attempted (and the goal is not even
type-correct at `implicit` transparency after unfolding).

The `reduceIndex*` dsimprocs instead navigate the constructor structure of the
`Index` expression themselves at default transparency, so no type-level matching is
needed. All reductions are definitional, hence `dsimproc` rather than `simproc`.
-/

open Lean Meta Simp in
/-- `reduce` at `default` transparency. Simprocs run at `reducible` transparency,
where `List.length`/`List.append` and friends do not unfold; the `Index`/`Curry`
computations below all live at the type level, so default transparency is needed. -/
private def Index.reduceD (e : Expr) : MetaM Expr :=
  withTransparency .default <| reduce e

open Lean Meta Simp in
/-- The length of a list expression built from `List.nil`/`List.cons`/`List.append`/
`List.map`/`List.replicate` (the list constructors arising in `Index` types). `whnf`
only reduces the head, so `List.length` cannot be evaluated directly. -/
private partial def Index.listLength? (γ : Expr) : MetaM (Option Nat) := do
  let γ ← whnfD γ
  let fn := γ.getAppFn
  let args := γ.getAppArgs
  if fn.isConstOf ``List.nil then
    return some 0
  else if fn.isConstOf ``List.cons && args.size == 3 then
    return (← listLength? args[2]!).map (· + 1)
  else if fn.isConstOf ``List.append && args.size == 3 then
    let some a ← listLength? args[1]! | return none
    let some b ← listLength? args[2]! | return none
    return some (a + b)
  else if fn.isConstOf ``List.map && args.size == 3 then
    listLength? args[2]!
  else if fn.isConstOf ``List.replicate && args.size == 3 then
    getNatValue? (← reduceD args[1]!)
  else
    return none

open Lean Meta Simp in
/-- If `e` is a `Fin` literal (an `OfNat` numeral, `Fin.mk` of a `Nat` literal,
or a `Fin.succ` chain on one), return its value. -/
private partial def Index.getFinVal? (e : Expr) : MetaM (Option Nat) := do
  if e.isAppOfArity ``Fin.succ 2 then
    return (← getFinVal? e.appArg!).map (· + 1)
  else if e.isAppOfArity ``Fin.mk 3 then
    getNatValue? (← reduceD e.getAppArgs[1]!)
  else
    -- `OfNat` numeral: the value is `k % n`, which is `k` for the literals that arise
    if let some (k, _) ← getOfNatValue? e ``Fin then return some k
    else return none

open Lean Meta Simp in
/-- Rebuild a stuck `Index` expression (a variable, or a constructor application that
is not navigated, e.g. `Index.select`) as an application to a literal `Fin`. -/
private def Index.stuckLeaf (idx : Expr) (k : Nat) : MetaM (Option Expr) := do
  match ← whnfD (← inferType idx) with
  | .forallE _ dom _ _ =>
    unless dom.isAppOfArity ``Fin 1 do return none
    let some n ← getNatValue? (← reduceD dom.appArg!) | return none
    if h : k < n then
      return some (mkApp idx (toExpr (⟨k, h⟩ : Fin n)))
    else
      return none
  | _ => return none

open Lean Meta Simp in
/--
Navigate the `Index` expression `idx` at (literal) position `k`, performing the
`Index.cons`/`Index.append`/`Index.single`/`Index.map`/`Index.unmap` reductions
structurally. `progress` tracks whether any constructor has been consumed, so that
rebuilding a stuck leaf only happens when the overall application simplifies.
-/
private partial def Index.nav (idx : Expr) (k : Nat) (progress : Bool) :
    MetaM (Option Expr) := do
  let fn := idx.getAppFn
  let args := idx.getAppArgs
  if fn.isConstOf ``Index.cons && args.size == 6 then
    if k == 0 then return some args[4]!
    else return ← nav args[5]! (k - 1) true
  else if fn.isConstOf ``Index.append && args.size == 6 then
    let some len ← listLength? args[2]! | return none
    if k < len then return ← nav args[4]! k true
    else return ← nav args[5]! (k - len) true
  else if fn.isConstOf ``Index.single && args.size == 4 then
    if k == 0 then return some args[3]! else return none
  else if (fn.isConstOf ``Index.map || fn.isConstOf ``Index.unmap) && args.size == 6 then
    return ← nav args[5]! k true
  else if fn.isConst || fn.isFVar then
    -- a head we do not navigate (a variable, `Index.null`, `Index.select`, ...):
    -- rebuilding the application is only progress if a constructor was consumed
    if progress then stuckLeaf idx k else return none
  else
    -- expose a constructor head through β-redexes, without unfolding definitions
    let idx' := idx.consumeMData.headBeta
    if idx' == idx then return none -- a `fun`/`match`: rebuilding would leave stuck matches
    else return ← nav idx' k progress

open Lean Meta Simp in
/-- Core of the `reduceIndex*` dsimprocs: `e` is an `Index` built from
`Index.cons`/`Index.append`/`Index.single`/`Index.map`/`Index.unmap`, applied to a
`Fin` position. If the position is a literal, reduce the application structurally. -/
private def Index.reduceCore (e : Expr) : SimpM DStep := do
  let i := e.appArg!
  let idx := e.appFn!
  let some k ← getFinVal? i | return .continue
  let some v ← nav idx k false | return .continue
  if v == e then return .continue
  return .visit v

/-- Reduce `Index.cons x₀ x i` at a literal position `i`. -/
dsimproc reduceIndexCons (Index.cons _ _ _) := Index.reduceCore

/-- Reduce `Index.append a b i` at a literal position `i`. -/
dsimproc reduceIndexAppend (Index.append _ _ _) := Index.reduceCore

/-- Reduce `Index.single x i` at a literal position `i`. -/
dsimproc reduceIndexSingle (Index.single _ _) := Index.reduceCore

/-- Reduce `Index.map a i` at a literal position `i`. -/
dsimproc reduceIndexMap (Index.map _ _) := Index.reduceCore

/-- Reduce `Index.unmap a i` at a literal position `i`. -/
dsimproc reduceIndexUnmap (Index.unmap _ _) := Index.reduceCore

attribute [reduce_soir] reduceIndexCons reduceIndexAppend reduceIndexSingle
  reduceIndexMap reduceIndexUnmap

open Lean Meta Simp in
/-- Core of `reduceCurryGet`: `e` is `Curry.get f i` with `i : Index m γ` for a
concrete list `γ`. Compute the get structurally:
`Curry.get f i = f (i 0) (i 1) ... (i (γ.length - 1))` (just `f` at `[]`).
The rewrite lemmas `Curry.get_zero`/`get_one`/`get_two`/... prove the same equations,
but only fire when `γ` is a literal cons-tower of length ≤ 3; this simproc works for
any list built from `nil`/`cons`/`append`/`map`/`replicate`, e.g. the `γ ++ γ'`
arising from `Curry.curry`/`Curry.transpose`. -/
private def Curry.reduceGetCore (e : Expr) : SimpM DStep := do
  let args := e.getAppArgs
  unless args.size == 6 do return .continue
  let γ := args[3]!
  let f := args[4]!
  let i := args[5]!
  let some len ← Index.listLength? γ | return .continue
  let mut v := f
  for j in [:len] do
    let some ij ← Index.stuckLeaf i j | return .continue
    v := mkApp v ij
  return .visit v

open Lean Meta Simp in
/-- Core of `reduceCurryOf`, unfolding one step of `Curry.of` on a concrete list
(`e` is an unapplied `Curry.of f`; applied occurrences are handled by rewriting the
unapplied `Curry.of f` subterm and β-reducing):
- at `[]`: rewrites to `f Index.null`;
- at `γ₀ :: γs`: rewrites to `fun v ↦ Curry.of fun a ↦ f (Index.cons v a)`, which is
  `Curry.of`'s definition on a cons, η-expanded. Iterating consumes one argument per
  step, ending at `f (Index.cons v₀ (... (Index.cons vₖ Index.null)))`.

The η-expansion would rewrite its own output forever, so it is skipped when `f`
already has the shape it produces (a lambda whose body ends in an `Index.cons`
applied to the bound variable). -/
private def Curry.reduceOfCore (e : Expr) : SimpM DStep := do
  let args := e.getAppArgs
  unless args.size == 5 do return .continue
  let some len ← Index.listLength? args[3]! | return .continue
  if len == 0 then
    return .visit (mkApp args[4]! (mkApp2 (mkConst ``Index.null []) args[0]! args[1]!))
  let f := args[4]!
  if f.isLambda then
    let body := f.bindingBody!
    if body.isApp && body.appArg!.getAppFn.isConstOf ``Index.cons
        && body.appArg!.appArg!.isBVar then
      return .continue
  let γ ← whnfD args[3]!
  let cargs := γ.getAppArgs
  unless γ.getAppFn.isConstOf ``List.cons && cargs.size == 3 do return .continue
  let dom := mkApp args[1]! cargs[1]!
  let idxTy ← mkAppM ``Index #[args[1]!, cargs[2]!]
  let r ← withLocalDecl `v .default dom fun v => do
    withLocalDecl `a .default idxTy fun a => do
      let cons ← mkAppM ``Index.cons #[v, a]
      let inner ← mkAppM ``Curry.of #[← mkLambdaFVars #[a] (mkApp f cons)]
      mkLambdaFVars #[v] inner
  return .visit r

/-- Reduce `Curry.get f i` structurally on concrete lists. -/
dsimproc reduceCurryGet (Curry.get _ _) := Curry.reduceGetCore

/-- Reduce `Curry.of f` structurally on concrete lists. -/
dsimproc reduceCurryOf (Curry.of _) := Curry.reduceOfCore

open Lean Meta Simp in
/-- Core of `reduceCurryArg`: `e` is `Curry.arg i` with `i : Fin γ.length` a literal
position in a concrete list `γ`. Rewrite to `Curry.of fun a ↦ a i` (definitionally
equal: both are the `i`-th projection η-expanded); `reduceCurryOf` and the
`reduceIndex*` simprocs then compute the projection structurally.

The rewrite lemmas `Curry.arg_zero`/`arg_one`/`arg_two`/`arg_succ` prove the same
equations, but only fire when the type of `i` is the unreduced `Fin γ.length`:
evaluation produces literals at the *reduced* type (e.g. `Fin 1`), which the lemmas
cannot match at `implicit` transparency. -/
private def Curry.reduceArgCore (e : Expr) : SimpM DStep := do
  let args := e.getAppArgs
  unless args.size == 4 do return .continue
  let γ := args[2]!
  let i := args[3]!
  let some k ← Index.getFinVal? i | return .continue
  let some len ← Index.listLength? γ | return .continue
  unless k < len do return .continue
  -- the element type `m γ[i]`, from the type `Curry m γ (m γ[i])` of `e`
  let ty ← inferType e
  unless ty.getAppFn.isConstOf ``Curry && ty.getAppArgs.size == 4 do return .continue
  let α := ty.getAppArgs[3]!
  let idxTy ← mkAppM ``Index #[args[1]!, γ]
  let g ← withLocalDecl `a .default idxTy fun a => mkLambdaFVars #[a] (mkApp a i)
  return .visit (mkAppN (mkConst ``Curry.of) #[args[0]!, args[1]!, α, γ, g])

/-- Reduce `Curry.arg i` at a literal position `i` in a concrete list. -/
dsimproc reduceCurryArg (Curry.arg _) := Curry.reduceArgCore

attribute [reduce_soir] reduceCurryGet reduceCurryOf reduceCurryArg

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

attribute [reduce_soir] Curry.transpose Curry.transposeFirst

end Soir
