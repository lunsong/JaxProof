import Xla

def power_loop_body {n : ℕ} : Xla.SimpleExpr [⟨.float, [n]⟩] ⟨.float, [n]⟩ :=
  Soir.Expr.ofFn fun x => Xla.mul x x

def power_loop {n : ℕ} : Xla.SimpleExpr [⟨.float, [n]⟩, ⟨.int, []⟩] ⟨.float, [n]⟩ :=
  Soir.Expr.ofFn fun x m => Xla.fori_loop power_loop_body m x .nil

#eval IO.println (power_loop (n := 10)).code

example (m n : ℕ) (x : Fin n → ℝ) :
    power_loop.eval x (m : ℤ) = x ^ (2 ^ m) := by
  simp [power_loop, reduce_xla, reduce_soir, power_loop_body, reduce_tensor]
  induction m with
  | zero => simp [Nat.repeat]
  | succ m ih =>
    simp only [Nat.repeat, reduce_soir]
    rw [ih, pow_succ, pow_mul, pow_two]
    rfl
