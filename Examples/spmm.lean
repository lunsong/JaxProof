import Xla

/-- Sparse-dense matrix product `A * B` in COO format: `rows`, `cols`, `vals`
describe the nonzero entries of `A : [n, m]`, and `b` is the dense right-hand
side `[m, l]`. Rows of `b` are gathered with the COO row indices, and a one-hot
matrix built from the COO column indices contracts the `nnz` axis, so duplicate
COO coordinates accumulate. -/
def spmm {n m l nnz : ℕ} :
  Xla.SimpleExpr
    [⟨.int, [nnz]⟩, ⟨.int, [nnz]⟩, ⟨.float, [nnz]⟩, ⟨.float, [m, l]⟩]
    ⟨.float, [n, l]⟩ :=
  Soir.Expr.ofFn fun rows cols vals b =>
    let rows_b := Xla.broadcast [⟨nnz, true⟩, ⟨l, false⟩] rows;
    let l_b := Xla.broadcast [⟨nnz, false⟩, ⟨l, true⟩] (Xla.iota l);
    let w := Xla.mul (Xla.gather b (rows_b.append l_b))
      (Xla.broadcast [⟨nnz, true⟩, ⟨l, false⟩] vals);
    let cols_b := Xla.broadcast [⟨nnz, true⟩, ⟨n, false⟩] cols;
    let n_b := Xla.broadcast [⟨nnz, false⟩, ⟨n, true⟩] (Xla.iota n);
    let p := Xla.choice (Xla.eq cols_b n_b) (Xla.ofNat 1) (Xla.ofNat 0);
    Xla.einsum [nnz, n, l] [[#0, #1], [#0, #2]] 1 (p.append w)

#eval IO.println (spmm (n := 3) (m := 4) (l := 2) (nnz := 3)).code
/-
%0 = broadcast [true, false]; $1
%1 = iota 3; 
%2 = broadcast [false, true]; %1
%3 = eq; %0, %2
%4 = const float [3, 3] 1; 
%5 = const float [3, 3] 0; 
%6 = where; %3, %4, %5
%7 = broadcast [true, false]; $0
%8 = iota 2; 
%9 = broadcast [false, true]; %8
%10 = gather; $3, %7, %9
%11 = broadcast [true, false]; $2
%12 = mul; %10, %11
%13 = einsum [[0, 1], [0, 2]] 1; %6, %12
return %13
-/

/-- Sparse matrix-vector product `A * x` in COO format: the vector analogue of
`spmm`, contracting the `nnz` axis with a one-hot matrix. -/
def spmv {n m nnz : ℕ} :
  Xla.SimpleExpr
    [⟨.int, [nnz]⟩, ⟨.int, [nnz]⟩, ⟨.float, [nnz]⟩, ⟨.float, [m]⟩]
    ⟨.float, [n]⟩ :=
  Soir.Expr.ofFn fun rows cols vals b =>
    let w := Xla.mul (Xla.gather b rows) vals;
    let cols_b := Xla.broadcast [⟨nnz, true⟩, ⟨n, false⟩] cols;
    let n_b := Xla.broadcast [⟨nnz, false⟩, ⟨n, true⟩] (Xla.iota n);
    let p := Xla.choice (Xla.eq cols_b n_b) (Xla.ofNat 1) (Xla.ofNat 0);
    Xla.einsum [nnz, n] [[#0, #1], [#0]] 1 (p.append w)

#eval IO.println (spmv (n := 3) (m := 4) (nnz := 3)).code
/-
%0 = broadcast [true, false]; $1
%1 = iota 3; 
%2 = broadcast [false, true]; %1
%3 = eq; %0, %2
%4 = const float [3, 3] 1; 
%5 = const float [3, 3] 0; 
%6 = where; %3, %4, %5
%7 = gather; $3, $0
%8 = mul; %7, $2
%9 = einsum [[0, 1], [0]] 1; %6, %8
return %9
-/

/-- Mathematical specification of `spmm`: entry `(i, j)` is the sum over all COO
entries whose column equals `i`, of the value times the gathered row of `b`
(row indices wrap modulo `m`). -/
def spmm_def {n m l nnz : ℕ} [NeZero m]
    (rows cols : Fin nnz → ℤ) (vals : Fin nnz → ℝ) (b : Xla.Tensor ℝ [m, l])
    (i : Fin n) (j : Fin l) : ℝ :=
  ∑ k, if cols k = (i.val : ℤ) then vals k * b (Fin.intCast (rows k) : Fin m) j else 0

theorem spmm_eq_def {n m l nnz : ℕ} [NeZero m]
    (rows cols : Fin nnz → ℤ) (vals : Fin nnz → ℝ) (b : Xla.Tensor ℝ [m, l])
    (i : Fin n) (j : Fin l) :
    spmm.eval rows cols vals b i j = spmm_def rows cols vals b i j := by
  sorry

/-- Mathematical specification of `spmv`: the vector analogue of `spmm_def`. -/
def spmv_def {n m nnz : ℕ} [NeZero m]
    (rows cols : Fin nnz → ℤ) (vals : Fin nnz → ℝ) (b : Xla.Tensor ℝ [m])
    (i : Fin n) : ℝ :=
  ∑ k, if cols k = (i.val : ℤ) then vals k * b (Fin.intCast (rows k) : Fin m) else 0

theorem spmv_eq_def {n m nnz : ℕ} [inst : NeZero m]
    (rows cols : Fin nnz → ℤ) (vals : Fin nnz → ℝ) (b : Xla.Tensor ℝ [m])
    (i : Fin n) :
    spmv.eval rows cols vals b i = spmv_def rows cols vals b i := by
  simp [spmv, reduce_xla, reduce_soir, reduce_tensor, spmv_def, inst.ne]
  erw [Finset.sum_apply]
  congr; ext k
  split_ifs
  · rw [mul_comm]
  · rfl
  
