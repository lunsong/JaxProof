import Xla

def norm_xla {n : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n]⟩]
    ⟨.float, []⟩ :=
  Soir.Expr.ofFn fun x ↦
    let x2 := Xla.mul x x;
    let x2_sumed := Xla.sum 1 x2;
    Xla.sqrt x2_sumed

def normalize_xla {n : ℕ} :
  Xla.SimpleExpr
    [⟨.float, [n]⟩]
    ⟨.float, [n]⟩ :=
  Soir.Expr.ofFn fun x ↦
    let x_norm := norm_xla.apply x;
    let x_norm_broadcasted := Xla.broadcast [⟨n, false⟩] x_norm;
    Xla.div x x_norm_broadcasted

#eval IO.println (normalize_xla (n := 12)).code

theorem norm_def (n : ℕ) (x : Fin n → ℝ) :
    norm_xla.eval x = √(∑ i, (x i)^2) := by
  simp [norm_xla, reduce_xla, reduce_soir, reduce_tensor, pow_two]
  rfl

theorem normalize_def (n : ℕ) (x : Fin n → ℝ) (i : Fin n) : 
    normalize_xla.eval x i = x i / √(∑ j, (x j)^2) := by
  simp [normalize_xla, reduce_xla, reduce_soir, reduce_tensor, id]
  congr
  exact norm_def n _
