import Xla

/-- Assemble a dense `[n, m]` matrix from a COO representation: index vectors
`rows`, `cols` and values `vals`, each of length `nnz`. The COO entries are
flattened row-major and fed to a 1-D `scatter`, so duplicate coordinates follow
`scatter`'s first-write-wins rule. -/
def cooToDense {n m nnz : ℕ} :
  Xla.SimpleExpr
    [⟨.int, [nnz]⟩, ⟨.int, [nnz]⟩, ⟨.float, [nnz]⟩]
    ⟨.float, [n, m]⟩ :=
  Soir.Expr.ofFn fun rows cols vals =>
    let flat := Xla.add (Xla.mul rows (Xla.ofNat m)) cols;
    Xla.unflatten [n, m] <|
      Xla.scatter (Xla.ofNat (out := ⟨.float, [(n :: m :: []).prod]⟩) 0) vals flat

#eval IO.println (cooToDense (n := 3) (m := 4) (nnz := 3)).code
/-
%0 = const float [12] 0; 
%1 = const int [3] 4; 
%2 = mul; $0, %1
%3 = add; %2, $1
%4 = scatter; %0, $2, %3
%5 = unflatten [3, 4]; %4
return %5
-/

/-- Mathematical specification of `cooToDense`: the entry `(i, j)` is the value of
the first COO entry whose (wrapped) coordinates are `(i, j)`, or `0` if there is none. -/
noncomputable def cooToDense_def {n m nnz : ℕ} [NeZero n] [NeZero m]
    (rows cols : Fin nnz → ℤ) (vals : Fin nnz → ℝ) (i : Fin n) (j : Fin m) : ℝ :=
  if h : ∃ k, (Fin.intCast (rows k) : Fin n) = i ∧ (Fin.intCast (cols k) : Fin m) = j
  then vals (Classical.choose h) else 0

theorem cooToDense_eq_def {n m nnz : ℕ} [NeZero n] [NeZero m]
    (rows cols : Fin nnz → ℤ) (vals : Fin nnz → ℝ)
    (hrows : ∀ k, 0 ≤ rows k ∧ rows k < n)
    (hcols : ∀ k, 0 ≤ cols k ∧ cols k < m)
    (huniq : Function.Injective fun k =>
      ((Fin.intCast (rows k) : Fin n), (Fin.intCast (cols k) : Fin m)))
    (i : Fin n) (j : Fin m) :
    cooToDense.eval rows cols vals i j = cooToDense_def rows cols vals i j := by
  sorry
