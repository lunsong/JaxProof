import Mathlib.NumberTheory.Divisors
import Mathlib.Data.Fin.VecNotation

example : Nat.divisors 35 = {1,5,7,35} := rfl

def foo (n : ℕ) : ℕ :=
  if n ≠ 0 ∧ n % 2 = 0 then foo (n / 2) else n

#check foo.eq_def
#check Acc.rec

def A' : List ℕ := [1,2,3,4]
def A : Fin 4 → ℕ := A'.get

