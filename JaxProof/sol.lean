inductive MyExpr (α : List Type) : Type → Type where
  | arg (i : Fin α.length) : MyExpr α (α.get i)

def MyExpr.toFin {α : List Type} {β : Type} : MyExpr α β → Fin α.length
  | arg i => i

theorem MyExpr.toFin.inj {α : List Type} {β : Type} {e₀ e₁ : MyExpr α β}
  (h : e₀.toFin = e₁.toFin) : e₀ = e₁ := by
  unfold MyExpr.toFin at h
  split at h
  split at h
  rename_i heq
  cases h
  cases heq
  rfl

instance MyExpr.instDecidableEq (α : List Type) (β : Type) : DecidableEq (MyExpr α β) :=
  fun x y =>
    if h : x.toFin = y.toFin then
      isTrue <| MyExpr.toFin.inj h
    else
      isFalse <| fun h' ↦ h (congrArg MyExpr.toFin h')



/-

theorem MyExpr.arg_toFin {argType : List Type} {exprType : Type}
  (expr : MyExpr argType exprType) : argType.get expr.toFin = exprType := by
  obtain ⟨i⟩ := expr
  rfl

#print Nat.succ.inj
#print Nat.casesOn
#print Nat.noConfusion
#print Nat.noConfusionType

theorem MyExpr.arg.inj (α : List Type) (i j : Fin α.length) (h : α.get i = α.get j) :
    MyExpr.arg α i = h ▸ (MyExpr.arg α j) → i = j := by
  generalize hx : MyExpr.arg α i = x
  generalize hβ : α.get i = β at 
  clear hx
  obtain ⟨j⟩ := x
-/

