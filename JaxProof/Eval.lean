import JaxProof.Expr

namespace Jax

def DList.append {α : Type} {γ : α → Type} {a b : List α}
  (x : DList γ a) (y : DList γ b) : DList γ (a ++ b) :=
  match x with
  | nil => y
  | cons x xs => cons x (xs.append y)

def DList.get {α : Type} {γ : α → Type} {a : List α}
    (i : Fin a.length) (x : DList γ a) : γ a[i] :=
  match a with
  | a :: as =>
    match x with
    | cons x xs =>
      match i with
      | .mk 0 _ => x
      | .mk (n + 1) h => xs.get <| .mk n <| by simpa using h

def DList.map {α : Type} {γ γ' : α → Type} {a : List α} (f : {a : α} → γ a → γ' a) :
    DList γ a → DList γ' a
  | nil => nil
  | cons x xs => cons (f x) (xs.map f)

class TensorImpl (tensor : TensorType → Type) where
  impl {args : List TensorType} {out : TensorType} : Op args out → DList tensor args → tensor out
  ofNat : ℕ → tensor ⟨.int, []⟩

def Expr.eval {args : List TensorType} {out : TensorType} (impl : TensorType → Type)
  [TensorImpl impl] (xs : DList impl args) (expr : Expr args out) : impl out :=
  match expr with
  | arg i => xs.get i
  --| bind op res => TensorImpl.impl op (res.map (Expr.eval impl xs))
  | bind op res => 
    let rec recursive_eval {ins : List TensorType} : DList (Expr args) ins → DList impl ins
    | .nil => .nil
    | .cons res res' => .cons (res.eval impl xs) (recursive_eval res')
    TensorImpl.impl op (recursive_eval res)
--  match args with
--  | [] => match expr with
--    | bind op xs => TensorImpl.impl op (xs.map (Expr.eval impl .nil))
--  | a :: as => sorry
--  expr.rec
--    (motive_1 := fun a e => impl a)
--    (motive_2 := fun a e => DList impl a)
--    (bind := fun op _ res => TensorImpl.impl op res)
--    (arg := xs.get)
--    (by sorry)
--    (by sorry)
--  | arg i => xs.get i
--  | bind op ins =>
--    TensorImpl.impl op (ins.map (Expr.eval impl xs))
    

def ExprGroup.eval {args outs : List TensorType} (impl : TensorType → Type) [TensorImpl impl]
    (xs : Tuple impl args) : ExprGroup args outs → Tuple impl outs
  | nil => ()
  | cons e es => ⟨e.eval impl xs, es.eval impl xs⟩
  | apply x f => f.eval impl (x.eval impl xs)
  | append x y => (x.eval impl xs).append (y.eval impl xs)
  | fori_loop (carry := carry) n step init aux =>
    let init := init.eval impl xs
    let aux := aux.eval impl xs
    let f i (a : Tuple impl carry) := (step.eval impl) ⟨TensorImpl.ofNat i, a.append aux⟩
    n.rec init f

end Jax
