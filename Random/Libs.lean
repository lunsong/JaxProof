import Random.Op
import Random.Meta

namespace Random

open Soir

variable {args ins outs : List RandType} {out : RandType}

@[reduce_random]
def bindPrim (op : RandPrimOp ins out) : Expr RandOp args ins → Expr RandOp args [out] :=
  fun xs => .bind (.simple op) (fun r => nomatch r) xs

@[reduce_random]
def add (x y : Expr RandOp args [.data]) : Expr RandOp args [.data] :=
  bindPrim .add (x.append y)

@[reduce_random]
def sub (x y : Expr RandOp args [.data]) : Expr RandOp args [.data] :=
  bindPrim .sub (x.append y)

@[reduce_random]
def mul (x y : Expr RandOp args [.data]) : Expr RandOp args [.data] :=
  bindPrim .mul (x.append y)

@[reduce_random]
def div (x y : Expr RandOp args [.data]) : Expr RandOp args [.data] :=
  bindPrim .div (x.append y)

@[reduce_random]
def neg (x : Expr RandOp args [.data]) : Expr RandOp args [.data] :=
  bindPrim .neg x

@[reduce_random]
def normal (k : Expr RandOp args [.key]) : Expr RandOp args [.data] :=
  bindPrim .normal k

@[reduce_random]
def key : Expr RandOp args [.key] :=
  bindPrim .key .nil

@[reduce_random]
def shuffle (k : Expr RandOp args [.key]) : Expr RandOp args [.key] :=
  bindPrim .shuffle k

instance : Add (Expr RandOp args [.data]) := ⟨add⟩

instance : Sub (Expr RandOp args [.data]) := ⟨sub⟩

instance : Mul (Expr RandOp args [.data]) := ⟨mul⟩

instance : Div (Expr RandOp args [.data]) := ⟨div⟩

instance : Neg (Expr RandOp args [.data]) := ⟨neg⟩

end Random
