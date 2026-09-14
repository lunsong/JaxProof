import Soir.Core

namespace Random

inductive RandType : Type where
  | data : RandType
  | key : RandType

inductive RandPrimOp : List RandType → RandType → Type where
  | ofNat : ℕ → RandPrimOp [] .data
  | add : RandPrimOp [.data, .data] .data
  | sub : RandPrimOp [.data, .data] .data
  | mul : RandPrimOp [.data, .data] .data
  | div : RandPrimOp [.data, .data] .data
  | neg : RandPrimOp [.data] .data
  | shuffle : RandPrimOp [.key] .key
  | normal : RandPrimOp [.key] .data
  | uniform : RandPrimOp [.key] .data

def RandPrimOp.toString {args : List RandType} {out : RandType} : RandPrimOp args out → String
  | ofNat n => s!"ofNat {n}"
  | add => "add"
  | sub => "sub"
  | mul => "mul"
  | div => "div"
  | neg => "neg"
  | normal => "normal"
  | uniform => "uniform"
  | shuffle => "shuffle"

instance (args : List RandType) (out : RandType) : ToString (RandPrimOp args out) :=
  ⟨RandPrimOp.toString⟩

abbrev RandOp : Soir.OpType RandType := Soir.SimpleOp RandPrimOp

end Random
