import SSA.Xla.Op

namespace Xla

open SSA

variable {args ins : List TensorType} {out : TensorType}

def bindPrim (op : XlaPrimOp ins out) : Expr XlaOp args ins → Expr XlaOp args [out] :=
  fun xs => .bind (.left (.simple op)) (fun r => nomatch r) xs

def transpose {α : DType} {s : Shape}
  (σ : Equiv.Perm (Fin s.length)) (x : Expr XlaOp args [⟨α, s⟩]) :
    Expr XlaOp args [⟨α, List.ofFn fun i => s[σ i]⟩] :=
  bindPrim (.transpose σ) x

def dot_general {α : DType} (batch contract lhs rhs : Shape)
  (x : Expr XlaOp args [⟨α, contract ++ batch ++ lhs⟩])
  (y : Expr XlaOp args [⟨α, contract ++ batch ++ rhs⟩]) :
    Expr XlaOp args [⟨α, batch ++ lhs ++ rhs⟩] :=
  bindPrim (.dot_general batch contract lhs rhs) (x.append y)

instance : Zero (Expr XlaOp args [out]) := ⟨bindPrim .zeros .nil⟩

def iota {n : ℕ} : Expr XlaOp args [⟨.int, [n]⟩] := bindPrim .iota .nil

instance (n : ℕ) : OfNat (Expr XlaOp args [out]) n := .mk <| bindPrim (.ofNat n) .nil

instance : Sub (Expr XlaOp args [out]) := .mk fun x y => bindPrim .sub (x.append y)

def cumsum (x : Expr XlaOp args [out]) : Expr XlaOp args [out] := bindPrim .cumsum x

def choice {α : DType} {s : Shape}
  (c : Expr XlaOp args [⟨.int, s⟩]) (x y : Expr XlaOp args [⟨α, s⟩]) : Expr XlaOp args [⟨α, s⟩] :=
  bindPrim .choice ((c.append x).append y)

def scatter {α : DType} {s : Shape} {n : ℕ}
  (x : Expr XlaOp args [⟨α, s⟩]) (y : Expr XlaOp args [⟨α, [n]⟩]) :
   Curry (fun ι ↦ Expr XlaOp args [ι])
   (List.replicate s.length ⟨.int, [n]⟩) (Expr XlaOp args [⟨α, s⟩]) :=
  Curry.of fun i =>
    bindPrim (.scatter (α := α) (s := s) (n := n)) <| (x.append y).append <| Expr.join.get i

def mul {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [σ] :=
  bindPrim .mul (x.append y)

def div {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [σ] :=
  bindPrim .div (x.append y)

def fori_loop {carry aux : List TensorType}
  (body : Expr XlaOp (carry ++ aux) carry)
  (n : Expr XlaOp args [TensorType.scalar .int])
  (init : Expr XlaOp args carry)
  (aux_val : Expr XlaOp args aux) :
  Expr XlaOp args carry :=
  Expr.bind (.right .repeat)
    (fun i => match i with | ⟨0, _⟩ => body)
    ((n.append init).append aux_val)

def sum {α : DType} {s : Shape} (n : ℕ) :
    Expr XlaOp args [⟨α, s⟩] → Expr XlaOp args [⟨α, s.drop n⟩] :=
  bindPrim (.sum n)


def broadcast {α : DType} (s : List (ℕ × Bool)) :
    Expr XlaOp args [⟨α, Tensor.preBroadcast s⟩] → Expr XlaOp args [⟨α, s.map Prod.fst⟩] :=
  bindPrim (.broadcast s)

def sqrt {s : Shape} : Expr XlaOp args [⟨.float, s⟩] → Expr XlaOp args [⟨.float, s⟩] :=
  bindPrim .sqrt

end Xla
