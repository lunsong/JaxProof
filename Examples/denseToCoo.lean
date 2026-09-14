import Xla

/-- 1-D nonzero rank: `nnzRank x i` is `0` when `x i = 0`, and otherwise the
number of nonzero entries among `x 0, ..., x i` (so the nonzero entries are
ranked `1, 2, ...` in order). -/
def nnzRank {N : ℕ} : Xla.SimpleExpr [⟨.float, [N]⟩] ⟨.int, [N]⟩ :=
  Soir.Expr.ofFn fun x =>
    let isZero := Xla.eq x (Xla.ofNat 0);
    let isNonzero := Xla.sub (Xla.ofNat 1) isZero;
    let rank := Xla.cumsum isNonzero;
    Xla.choice isNonzero rank 0

#eval IO.println (nnzRank (N := 6)).code
/-
%0 = const int [6] 1; 
%1 = const float [6] 0; 
%2 = eq; $0, %1
%3 = sub; %0, %2
%4 = cumsum; %3
%5 = zeros int [6]; 
%6 = where; %3, %4, %5
return %6
-/

/-- Compacted COO values of a dense `[n, m]` matrix: entry `r` is the `(r+1)`-th
nonzero value in row-major order, or `0` past the nonzero count. -/
def cooVals {n m : ℕ} : Xla.SimpleExpr [⟨.float, [n, m]⟩] ⟨.float, [n * m]⟩ :=
  Soir.Expr.ofFn fun x =>
    let flat := Xla.flatten x;
    let pos := Xla.sub (nnzRank.apply flat) (Xla.ofNat 1);
    Xla.cast (show [(n :: m :: []).prod] = [n * m] by simp)
      (Xla.scatter (Xla.ofNat (out := ⟨.float, [(n :: m :: []).prod]⟩) 0) flat pos)

/-- Compacted COO row indices of a dense `[n, m]` matrix: entry `r` is the row of
the `(r+1)`-th nonzero element in row-major order, or `0` past the count. -/
def cooRows {n m : ℕ} : Xla.SimpleExpr [⟨.float, [n, m]⟩] ⟨.int, [n * m]⟩ :=
  Soir.Expr.ofFn fun x =>
    let rowOf := Xla.broadcast [⟨n, true⟩, ⟨m, false⟩] (Xla.iota n);
    let pos := Xla.sub (nnzRank.apply (Xla.flatten x)) (Xla.ofNat 1);
    Xla.cast (show [(n :: m :: []).prod] = [n * m] by simp)
      (Xla.scatter (Xla.ofNat (out := ⟨.int, [(n :: m :: []).prod]⟩) 0)
        (Xla.flatten rowOf) pos)

/-- Compacted COO column indices of a dense `[n, m]` matrix: entry `r` is the
column of the `(r+1)`-th nonzero element in row-major order, or `0` past the
count. -/
def cooCols {n m : ℕ} : Xla.SimpleExpr [⟨.float, [n, m]⟩] ⟨.int, [n * m]⟩ :=
  Soir.Expr.ofFn fun x =>
    let colOf := Xla.broadcast [⟨n, false⟩, ⟨m, true⟩] (Xla.iota m);
    let pos := Xla.sub (nnzRank.apply (Xla.flatten x)) (Xla.ofNat 1);
    Xla.cast (show [(n :: m :: []).prod] = [n * m] by simp)
      (Xla.scatter (Xla.ofNat (out := ⟨.int, [(n :: m :: []).prod]⟩) 0)
        (Xla.flatten colOf) pos)

/-- Dense-to-COO conversion: the COO row indices, column indices and values of a
dense `[n, m]` matrix, each of capacity `n * m`, with the nonzeros compacted to
the front in row-major order and `0` padding afterwards. -/
def denseToCoo {n m : ℕ} :
    Soir.Expr Xla.XlaOp [⟨.float, [n, m]⟩]
      [⟨.int, [n * m]⟩, ⟨.int, [n * m]⟩, ⟨.float, [n * m]⟩] :=
  Soir.Expr.ofFn fun x =>
    (cooRows.apply x).append ((cooCols.apply x).append (cooVals.apply x))

#eval IO.println (denseToCoo (n := 3) (m := 4)).code
/-
%0 = call; @1, $0
%1 = call; @2, $0
%2 = call; @3, $0
return %0,%1,%2

@0:
%0 = const int [12] 1; 
%1 = const float [12] 0; 
%2 = eq; $0, %1
%3 = sub; %0, %2
%4 = cumsum; %3
%5 = zeros int [12]; 
%6 = where; %3, %4, %5
return %6

@1:
%0 = const int [12] 0; 
%1 = iota 3; 
%2 = broadcast [true, false]; %1
%3 = flatten; %2
%4 = flatten; $0
%5 = call; @0, %4
%6 = const int [12] 1; 
%7 = sub; %5, %6
%8 = scatter; %0, %3, %7
%9 = id; %8
return %9

@2:
%0 = const int [12] 0; 
%1 = iota 4; 
%2 = broadcast [false, true]; %1
%3 = flatten; %2
%4 = flatten; $0
%5 = call; @0, %4
%6 = const int [12] 1; 
%7 = sub; %5, %6
%8 = scatter; %0, %3, %7
%9 = id; %8
return %9

@3:
%0 = const float [12] 0; 
%1 = flatten; $0
%2 = call; @0, %1
%3 = const int [12] 1; 
%4 = sub; %2, %3
%5 = scatter; %0, %1, %4
%6 = id; %5
return %6
-/

/-- Mathematical specification of `nnzRank`. -/
noncomputable def nnzRank_def {N : ℕ} (x : Fin N → ℝ) (i : Fin N) : ℤ :=
  if x i = 0 then 0
  else ((Finset.univ.filter fun j : Fin N => j ≤ i ∧ x j ≠ 0).card : ℤ)

theorem nnzRank_eq_def {N : ℕ} (x : Fin N → ℝ) (i : Fin N) :
    nnzRank.eval x i = nnzRank_def x i := by
  sorry

/-- Mathematical specification of `cooVals`: slot `r` holds the value of the
`(r+1)`-th nonzero in row-major order, or `0` if there is none. -/
noncomputable def cooVals_def {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (r : Fin (n * m)) : ℝ :=
  if h : ∃ k : Fin (List.prod [n, m]),
      Xla.Tensor.flatten x k ≠ 0 ∧ nnzRank_def (Xla.Tensor.flatten x) k = (r.val : ℤ) + 1
  then Xla.Tensor.flatten x (Classical.choose h) else 0

/-- Mathematical specification of `cooRows`: slot `r` holds the row of the
`(r+1)`-th nonzero in row-major order, or `0` if there is none. -/
noncomputable def cooRows_def {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (r : Fin (n * m)) : ℤ :=
  if h : ∃ k : Fin (List.prod [n, m]),
      Xla.Tensor.flatten x k ≠ 0 ∧ nnzRank_def (Xla.Tensor.flatten x) k = (r.val : ℤ) + 1
  then ((Classical.choose h).divNat : Fin n).val else 0

/-- Mathematical specification of `cooCols`: slot `r` holds the column of the
`(r+1)`-th nonzero in row-major order, or `0` if there is none. -/
noncomputable def cooCols_def {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (r : Fin (n * m)) : ℤ :=
  if h : ∃ k : Fin (List.prod [n, m]),
      Xla.Tensor.flatten x k ≠ 0 ∧ nnzRank_def (Xla.Tensor.flatten x) k = (r.val : ℤ) + 1
  then ((Classical.choose h).modNat : Fin (m * 1)).val else 0

theorem cooVals_eq_def {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (r : Fin (n * m)) :
    cooVals.eval x r = cooVals_def x r := by
  sorry

theorem cooRows_eq_def {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (r : Fin (n * m)) :
    cooRows.eval x r = cooRows_def x r := by
  sorry

theorem cooCols_eq_def {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) (r : Fin (n * m)) :
    cooCols.eval x r = cooCols_def x r := by
  sorry

/-- The three outputs of `denseToCoo` are the independently specified COO arrays. -/
theorem denseToCoo_parts {n m : ℕ} (x : Xla.Tensor ℝ [n, m]) :
    (Soir.Expr.eval Xla.DirectImpl denseToCoo) x 0 = cooRows.eval x ∧
    (Soir.Expr.eval Xla.DirectImpl denseToCoo) x 1 = cooCols.eval x ∧
    (Soir.Expr.eval Xla.DirectImpl denseToCoo) x 2 = cooVals.eval x := by
  sorry
