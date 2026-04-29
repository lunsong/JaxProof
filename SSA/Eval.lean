import JaxProof.Expr

namespace Jax

def TensorImpl.implType
  (tensorImpl : TensorType → Type) (args : List TensorType) (out : TensorType) : Type :=
  match args with
  | [] => tensorImpl out
  | arg :: args => tensorImpl arg → implType tensorImpl args out

class TensorImpl
  (opType : List TensorType → TensorType → Type)
  (tensorImpl : TensorType → Type) where
  impl {args : List TensorType} {out : TensorType} :
    opType args out → TensorImpl.implType tensorImpl args out
  toInt : tensorImpl TensorType.int_scalar → ℤ
  ofNat : ℕ → tensorImpl ⟨.int, []⟩

abbrev DList (impl : TensorType → Type) (a : List TensorType) : Type :=
  ∀ i : Fin a.length, impl a[i]

def Expr.evalType (impl : TensorType → Type) (args outs : List TensorType) : Type :=
  match args with
  | [] => ∀ i : Fin outs.length, impl outs[i]
  | a :: args => impl a → evalType impl args outs

def Expr.evalType.const {impl : TensorType → Type} {args outs : List TensorType}
  (x : ∀ i : Fin outs.length, impl outs[i]) : evalType impl args outs :=
  match args with
  | [] => x
  | _ :: _ => fun _ => const x

def Expr.evalType.arg {impl : TensorType → Type} {args : List TensorType} (i : Fin args.length) :
    evalType impl args [args[i]] :=
  match args with
  | a :: as =>
    match i with
    | 0 => fun x => const fun r => match r with | 0 => x
    | .mk (i + 1) hi => fun _ => arg <| Fin.mk i <| by simpa using hi

def Expr.evalType.toFin {impl : TensorType → Type} {args outs : List TensorType} :
    evalType impl args outs → DList impl args → DList impl outs :=
  match args with
  | [] => fun x _ => x
  | _ :: _ => fun f x => (f (x 0)).toFin <| fun i => x i.succ

def Expr.evalType.ofFin {impl : TensorType → Type} {args outs : List TensorType} :
    (DList impl args → DList impl outs) → evalType impl args outs :=
  match args with
  | [] => fun x => x fun r => nomatch r
  | _ :: _ => fun x a => ofFin fun u => x fun r =>
    match r with
    | 0 => a
    | .mk (r + 1) hr => u <| .mk r <| by simpa using hr

def Expr.evalType.ofOp {impl : TensorType → Type} {args : List TensorType} {out : TensorType} :
    TensorImpl.implType impl args out → evalType impl args [out] :=
  match args with
  | [] => fun op r =>
    match r with
    | 0 => op
  | _ :: _ => fun op x => ofOp <| op x

def Expr.evalType.append {impl : TensorType → Type} {a b : List TensorType}
  (x : DList impl a) (y : DList impl b) : DList impl (a ++ b) := 
  match a with
  | [] => y
  | a₀ :: as => fun i =>
    match i with
    | .mk 0 h => x 0
    | .mk (i + 1) h =>
      let x : ∀ i : Fin as.length, impl as[i] := fun i => x i.succ
      append x y <| .mk i <| by simpa using h

def Expr.evalType.select {impl : TensorType → Type} {a : List TensorType}
  (is : List (Fin a.length)) (x : DList impl a) : DList impl (is.map a.get) :=
  match is with
  | [] => fun i => nomatch i
  | i₀ :: is => fun i =>
    match i with
    | .mk 0 h => x i₀
    | .mk (i + 1) h =>
      select is x <| .mk i <| by simpa using h

def Expr.eval {opType : List TensorType → TensorType → Type} {args outs : List TensorType}
  (impl : TensorType → Type) [TensorImpl opType impl] (expr : Expr opType args outs) : 
    evalType impl args outs := 
  match expr with
  | nil => evalType.const fun r => nomatch r
  | arg i => evalType.arg i
  | bind op x =>
    let op := evalType.toFin <| evalType.ofOp <| TensorImpl.impl op
    let x := (x.eval impl).toFin
    evalType.ofFin (op ∘ x)
  | apply f g =>
    evalType.ofFin ((f.eval impl).toFin ∘ (g.eval impl).toFin)
  | append a b =>
    evalType.ofFin fun r =>
      let a := (a.eval impl).toFin r
      let b := (b.eval impl).toFin r
      evalType.append a b
  | select i xs =>
    evalType.ofFin fun r =>
      let xs := (xs.eval impl).toFin r
      evalType.select i xs
  | fori_loop (carry := carry) n f init =>
    evalType.ofFin fun x =>
      let init := (init.eval impl).toFin x
      let f : ℕ → DList impl carry → DList impl carry :=
        fun i a => (f.eval impl (TensorImpl.ofNat opType i)).toFin a
      Nat.rec init f n

end Jax
