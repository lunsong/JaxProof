import Soir.Core

namespace Random

/-- Data universe of the Random dialect: ordinary float scalars, probability
measures over the reals, and pseudo-random number generator keys. -/
inductive DataType : Type where
  | real : DataType
  | random : DataType

def DataType.max : DataType → DataType → DataType
  | .random, _ => .random
  | _, .random => .random
  | .real, .real => .real

inductive RandType : Type where
  | data : DataType → RandType
  | key : RandType

instance : ToString RandType where
  toString
  | .data .real => "real"
  | .data .random => "random"
  | .key => "key"

inductive RandPrimOp : List RandType → RandType → Type where
  | add {α β : DataType} : RandPrimOp [.data α, .data α] (.data (DataType.max α β))
  | sub {α β : DataType} : RandPrimOp [.data α, .data α] (.data (DataType.max α β))
  | mul {α β : DataType} : RandPrimOp [.data α, .data α] (.data (DataType.max α β))
  | div {α β : DataType} : RandPrimOp [.data α, .data α] (.data (DataType.max α β))
  | neg {α : DataType} : RandPrimOp [.data α] (.data α)
  | key : RandPrimOp [] .key
  | shuffle : RandPrimOp [.key] .key
  | normal : RandPrimOp [.key] (.data .random)

def RandPrimOp.toString {args : List RandType} {out : RandType} : RandPrimOp args out → String
  | add => "add"
  | sub => "sub"
  | mul => "mul"
  | div => "div"
  | neg => "neg"
  | normal => "normal"
  | key => "key"
  | shuffle => "shuffle"

instance (args : List RandType) (out : RandType) : ToString (RandPrimOp args out) :=
  ⟨RandPrimOp.toString⟩

abbrev RandOp : Soir.OpType RandType := Soir.SimpleOp RandPrimOp

end Random
