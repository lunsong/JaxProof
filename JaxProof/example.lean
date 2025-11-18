inductive MyExpr (argType : List Type) : Type → Type 1 where
  | arg (i : Fin argType.length) : MyExpr argType argType[i]
  | const {exprType : Type} : exprType → MyExpr argType exprType
--deriving DecidableEq

def MyExpr.eval {argType : List Type} {exprType : Type} (expr : MyExpr argType exprType):
    (∀ i : Fin argType.length, argType[i]) → exprType :=
  match expr with
  | MyExpr.arg i => fun x => x i
  | MyExpr.const x => fun _ => x

