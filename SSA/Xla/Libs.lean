import SSA.Xla.Op

namespace Xla

open SSA

variable {args ins outs : List TensorType} {out : TensorType}

def bindPrim (op : XlaPrimOp ins out) : Expr XlaOp args ins → Expr XlaOp args [out] :=
  fun xs => .bind (.left (.simple op)) (fun r => nomatch r) xs

def makePerm {n : ℕ} (x : Fin n → Fin n) (h1 : Function.Surjective x) (h2 : Function.Injective x) :
    Equiv.Perm (Fin n) where
  toFun := x
  invFun i := Fin.find (fun j => x j = i) (h1 i)
  right_inv i := by simp [Fin.find_spec (h1 i)]
  left_inv i := by simp only [Fin.find_eq_iff, true_and]; intro j hj hc; exact hj.ne (h2 hc)

def transpose {α : DType} {s : Shape}
  (σ : Equiv.Perm (Fin s.length)) (x : Expr XlaOp args [⟨α, s⟩]) :
    Expr XlaOp args [⟨α, List.ofFn fun i => s[σ i]⟩] :=
  bindPrim (.transpose σ) x

def transpose' {α : DType} {s : Shape}
  (x : Expr XlaOp args [⟨α, s⟩])
  (σ : Fin s.length → Fin s.length)
  (h1 : Function.Surjective σ := by simp; decide)
  (h2 : Function.Injective σ := by simp; decide) :
    Expr XlaOp args [⟨α, List.ofFn fun i => s[σ i]⟩] :=
  bindPrim (.transpose (makePerm σ h1 h2)) x

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

def exp {s : Shape} : Expr XlaOp args [⟨.float, s⟩] → Expr XlaOp args [⟨.float, s⟩] :=
  bindPrim .exp

def neg : Expr XlaOp args [out] → Expr XlaOp args [out] := bindPrim .neg

def choice {α : DType} {s : Shape}
  (c : Expr XlaOp args [⟨.int, s⟩]) (x y : Expr XlaOp args [⟨α, s⟩]) : Expr XlaOp args [⟨α, s⟩] :=
  bindPrim .choice ((c.append x).append y)

def scatter {α : DType} {s : Shape} {n : ℕ}
  (x : Expr XlaOp args [⟨α, s⟩]) (y : Expr XlaOp args [⟨α, [n]⟩]) :
   Curry (fun ι ↦ Expr XlaOp args [ι])
   (List.replicate s.length ⟨.int, [n]⟩) (Expr XlaOp args [⟨α, s⟩]) :=
  Curry.of fun i =>
    bindPrim (.scatter (α := α) (s := s) (n := n)) <| (x.append y).append <| Expr.join.get i

def sub {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [σ] :=
  bindPrim .sub (x.append y)

def add {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [σ] :=
  bindPrim .add (x.append y)

def mul {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [σ] :=
  bindPrim .mul (x.append y)

def mod {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [σ] :=
  bindPrim .mod (x.append y)

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

def vmap (batch : ℕ) {aux : List TensorType}
  (fn : Expr XlaOp (ins ++ aux) outs)
  (xs : Expr XlaOp args (ins.map fun ⟨α, s⟩ ↦ ⟨α, batch :: s⟩))
  (aux : Expr XlaOp args aux) :
    Expr XlaOp args (outs.map fun ⟨α, s⟩ ↦ ⟨α, batch :: s⟩) :=
  Expr.bind (.right .vmap)
    (fun i => match i with | ⟨0, _⟩ => fn)
    (xs.append aux)

/-
def vmap₂ (batch₀ batch₁ : ℕ) (ins : List TensorType) {aux : List TensorType}
  (fn : Expr XlaOp (ins ++ aux) outs)
  (xs : Expr XlaOp args (ins.map fun ⟨α, s⟩ ↦ ⟨α, batch₀ :: batch₁ :: s⟩))
  (aux : Expr XlaOp args aux) :
    Expr XlaOp args (outs.map fun ⟨α, s⟩ ↦ ⟨α, batch₀ :: batch₁ :: s⟩) :=
  let fn := vmap batch₁ fn (Expr.ofAppend' Expr.id) (Expr.ofAppend Expr.id)
  sorry
  --vmap batch₀ fn xs aux
-/

def sum {α : DType} {s : Shape} (n : ℕ) :
    Expr XlaOp args [⟨α, s⟩] → Expr XlaOp args [⟨α, s.drop n⟩] :=
  bindPrim (.sum n)

def broadcast {α : DType} (s : List (ℕ × Bool)) :
    Expr XlaOp args [⟨α, Tensor.preBroadcast s⟩] → Expr XlaOp args [⟨α, s.map Prod.fst⟩] :=
  bindPrim (.broadcast s)

def sqrt {s : Shape} : Expr XlaOp args [⟨.float, s⟩] → Expr XlaOp args [⟨.float, s⟩] :=
  bindPrim .sqrt

def einsum (s : Shape) (indices : List (List (Fin s.length))) (n : ℕ)
  (xs : Expr XlaOp args (indices.map fun i ↦ ⟨.float, i.map s.get⟩)) :
    Expr XlaOp args [⟨.float, s.drop n⟩] :=
  bindPrim (.einsum s indices n) xs

def gather {α : DType} {s s' : Shape}
  (x : Expr XlaOp args [⟨α, s⟩])
  (indices : Expr XlaOp args (List.replicate s.length ⟨.int, s'⟩)) :
    Expr XlaOp args [⟨α, s'⟩] :=
  bindPrim .gather (x.append indices)

def eq {σ : TensorType} (x y : Expr XlaOp args [σ]) : Expr XlaOp args [⟨.int, σ.shape⟩] :=
  bindPrim .eq (x.append y)

end Xla
