import JaxProof.Expr

namespace Jax

def Tuple (impl : TensorType → Type) : List TensorType → Type
  | [] => Unit
  | σ :: σs => impl σ × Tuple impl σs

def Tuple.append {impl : TensorType → Type} {x y : List TensorType}
  (a : Tuple impl x) (b : Tuple impl y) : Tuple impl (x ++ y) :=
  match x with
  | [] => b
  | _ :: _ =>
    let ⟨a, as⟩ := a
    ⟨a, as.append b⟩

def Tuple.get {impl : TensorType → Type} {args : List TensorType} (i : Fin args.length) :
    Tuple impl args → impl args[i] := 
  match args with
  | arg :: args =>
    fun ⟨x, xs⟩ ↦
    match i with
    | 0 => x
    | .mk (n + 1) h => xs.get <| .mk n <| by simpa using h
    

class TensorImpl (tensor : TensorType → Type) where
  impl {args : List TensorType} {out : TensorType} : Op args out → Tuple tensor args → tensor out
  ofNat : ℕ → tensor ⟨.int, []⟩

def Expr.eval {args : List TensorType} {out : TensorType} (impl : TensorType → Type)
  [TensorImpl impl] (xs : Tuple impl args) : Expr args out → impl out
  | arg i => xs.get i
  | nullop op => TensorImpl.impl op ()
  | unop op x => TensorImpl.impl op ⟨x.eval impl xs, ()⟩
  | binop op x y => TensorImpl.impl op ⟨x.eval impl xs, y.eval impl xs, ()⟩
    

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
