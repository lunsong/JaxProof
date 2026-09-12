import Xla

def softmax {n : ℕ} : Xla.SimpleExpr [⟨.float, [n]⟩] ⟨.float, [n]⟩ :=
  Soir.Expr.ofFn fun x =>
  let exp_x := Xla.exp x;
  let sum_exp_x := Xla.broadcast [⟨n, false⟩] (Xla.sum 1 exp_x);
  Xla.div exp_x sum_exp_x

noncomputable def softmax_def {n : ℕ} (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (x i) / ∑ i, Real.exp (x i)

theorem softmax_eq_def {n : ℕ} :
    (softmax (n := n)).eval = softmax_def := by
  ext x i
  simp [softmax, reduce_xla, reduce_soir, reduce_tensor]
  rfl

def attention {N_key N_feat_in N_feat_out : ℕ} :
  Xla.SimpleExpr
    [
      ⟨.float, [N_feat_in]⟩,
      ⟨.float, [N_key, N_feat_in]⟩,
      ⟨.float, [N_key, N_feat_out]⟩
    ]
    ⟨.float, [N_feat_out]⟩ :=
  Soir.Expr.ofFn fun q k v =>
    let w := Xla.einsum [N_feat_in, N_key] [[#0], [#1, #0]] 1 (q.append k);
    let w := softmax.apply w;
    Xla.einsum [N_key, N_feat_out] [[#0], [#0, #1]] 1 (w.append v)

noncomputable def attention_def {N_key N_feat_in N_feat_out : ℕ}
  (q : Fin N_feat_in → ℝ)
  (k : Fin N_key → Fin N_feat_in → ℝ)
  (v : Fin N_key → Fin N_feat_out → ℝ)
  (i : Fin N_feat_out) : ℝ :=
  let w : Fin N_key → ℝ := fun a => ∑ j, q j * k a j
  ∑ a, softmax_def w a * v a i

theorem attention_eq_def {N_key N_feat_in N_feat_out : ℕ} :
    (@attention N_key N_feat_in N_feat_out).eval = attention_def := by
  ext q k v i
  simp [attention, reduce_xla, reduce_soir, reduce_tensor, attention_def]
  erw [Finset.sum_apply]
  congr
  ext j
  congr
  rw [← softmax_eq_def]
  simp [Xla.SimpleExpr.eval, Soir.Curry.map]
  apply congrFun
  apply congrFun
  apply congrArg
  ext a
  erw [Finset.sum_apply]
  
#eval IO.println (@attention 10 20 30).code
