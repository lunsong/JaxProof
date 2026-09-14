import Random.Op
import Random.Meta

namespace Random

open Soir

variable {args ins outs : List RandType} {out : RandType}

@[reduce_random]
def bindPrim (op : RandPrimOp ins out) : Expr RandOp args ins → Expr RandOp args [out] :=
  fun xs => .bind (.simple op) (fun r => nomatch r) xs

@[reduce_random]
def add (x y : Expr RandOp args [.float]) : Expr RandOp args [.float] :=
  bindPrim .add (x.append y)

@[reduce_random]
def sub (x y : Expr RandOp args [.float]) : Expr RandOp args [.float] :=
  bindPrim .sub (x.append y)

@[reduce_random]
def mul (x y : Expr RandOp args [.float]) : Expr RandOp args [.float] :=
  bindPrim .mul (x.append y)

@[reduce_random]
def div (x y : Expr RandOp args [.float]) : Expr RandOp args [.float] :=
  bindPrim .div (x.append y)

@[reduce_random]
def neg (x : Expr RandOp args [.float]) : Expr RandOp args [.float] :=
  bindPrim .neg x

@[reduce_random]
def normal (k : Expr RandOp args [.key]) : Expr RandOp args [.measure] :=
  bindPrim .normal k

@[reduce_random]
def key : Expr RandOp args [.key] :=
  bindPrim .key .nil

@[reduce_random]
def shuffle (k : Expr RandOp args [.key]) : Expr RandOp args [.key] :=
  bindPrim .shuffle k

/-- Joint law of two measures: `prod x y` is a measure over the concatenated
coordinates of `x` and `y`. -/
@[reduce_random]
def prod (x y : Expr RandOp args [.measure]) : Expr RandOp args [.measure] :=
  bindPrim .prod (x.append y)

instance : Add (Expr RandOp args [.float]) := ⟨add⟩

instance : Sub (Expr RandOp args [.float]) := ⟨sub⟩

instance : Mul (Expr RandOp args [.float]) := ⟨mul⟩

instance : Div (Expr RandOp args [.float]) := ⟨div⟩

instance : Neg (Expr RandOp args [.float]) := ⟨neg⟩

/-- `+` on measures denotes the joint (product) law. -/
instance : Add (Expr RandOp args [.measure]) := ⟨prod⟩

end Random
