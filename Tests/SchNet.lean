import SSA

/-
def SchNet (n_species embed_dim n_atom : ℕ) :=
  ssa Xla.XlaOp with
    Z : ⟨.int, [n_atom]⟩,
    x : ⟨.float, [n_atom, 3]⟩,
    θ₀ : ⟨.float, [embed
-/

def embed (n_species embed_dim : ℕ) :=
  ssa Xla.XlaOp with
    x : ⟨.int, []⟩,
    θ : ⟨.float, [n_species, embed_dim]⟩
  begin
    let x := Xla.broadcast [⟨embed_dim, false⟩] x;
    let_expr i : [⟨.int, [embed_dim]⟩] := Xla.iota;
    let y := Xla.gather θ (x.append i);
    return y

def embed_def (n_species embed_dim : ℕ) [NeZero n_species]
  (x : ℤ) (θ : Fin n_species → Fin embed_dim → ℝ) : Fin embed_dim → ℝ :=
  θ (Fin.intCast x)

theorem embed_eq_def (n_species embed_dim : ℕ) [inst : NeZero n_species] :
    Xla.simpleEval (embed n_species embed_dim) = embed_def n_species embed_dim := by
  ext x θ
  simp only [List.map_cons, List.map_nil, Xla.simpleEval, Curry.map, embed, Xla.gather,
    Xla.bindPrim, List.length_cons, List.length_nil, Nat.reduceAdd, List.reduceReplicate,
    Fin.getElem_fin, Fin.mk_one, Fin.isValue, List.replicate_zero, Xla.broadcast, Fin.zero_eta,
    Xla.iota, SSA.Expr.eval, Curry.get, SSA.evalType.bind, SSA.Impl.bind, Index.single,
    SSA.SimpleImpl.bind, Xla.DirectImpl.gather, List.replicate_one, Curry.of, inst.out, ↓reduceDIte,
    Curry.pure, Curry.map₂, Index.append, Curry.arg, Fin.succ_zero_eq_one, SSA.Tensor.broadcast,
    id_eq, Fin.succ_one_eq_two, List.nil_append, embed_def]
  ext r
  simp [ne_zero_of_lt r.isLt, Fin.intCast]
  rfl

def pairwise_offset {n_atom : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n_atom, 3]⟩
  begin
    let u := Xla.broadcast [⟨n_atom, true⟩, ⟨n_atom, false⟩, ⟨3, true⟩] x;
    let v := Xla.broadcast [⟨n_atom, false⟩, ⟨n_atom, true⟩, ⟨3, true⟩] x;
    return Xla.sub u v

def pairwise_offset_def {n_atom : ℕ}
  (x : Fin n_atom → Fin 3 → ℝ) (i j : Fin n_atom) (k : Fin 3) : ℝ :=
  x i k - x j k

theorem pairwise_offset_eq_def {n_atom : ℕ} :
    Xla.simpleEval (pairwise_offset (n_atom := n_atom)) = pairwise_offset_def := by
  ext x
  rfl

def periodic_distance :=
  ssa Xla.XlaOp with
    x : ⟨.float, [3]⟩,
    lattice : ⟨.float, []⟩
  begin
    let lattice := Xla.broadcast [⟨3, false⟩] lattice;
    let half_lattice := Xla.div lattice 2;
    let x := Xla.add x half_lattice;
    let x := Xla.mod x lattice;
    let x := Xla.sub x half_lattice;
    return Xla.sum 1 (Xla.mul x x)

noncomputable def periodic_distance_def (x : Fin 3 → ℝ) (lattice : ℝ) : ℝ :=
  ∑ i, ((x i + lattice / 2) % lattice - lattice / 2) ^ 2

theorem periodic_distance_eq_def : Xla.simpleEval periodic_distance = periodic_distance_def := by
  ext x L
  simp only [OfNat.ofNat, List.drop_succ_cons, List.drop_zero, Xla.simpleEval, Curry.map,
    periodic_distance, Xla.sum, Xla.bindPrim, List.length_nil, Fin.getElem_fin, Xla.mul, Xla.sub,
    Xla.mod, Xla.add, List.length_cons, Nat.reduceAdd, Fin.zero_eta, Fin.isValue, Xla.div,
    List.map_cons, List.map_nil, Xla.broadcast, Fin.mk_one, SSA.Expr.eval, Curry.get,
    SSA.evalType.bind, SSA.Impl.bind, Index.single, SSA.SimpleImpl.bind, SSA.Tensor.sumN,
    SSA.Tensor.sumFirst, SSA.Tensor.map₂, Curry.map₂, Index.append, Curry.arg, Curry.pure,
    Fin.succ_zero_eq_one, SSA.Tensor.broadcast, id_eq, List.nil_append, List.tail_cons,
    periodic_distance_def]
  conv_rhs =>
    arg 2; intro i
    rw [pow_two]
  

def periodic_distance_vmapped {n_atom : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n_atom, 3]⟩,
    lattice : ⟨.float, []⟩
  begin
    return Xla.vmap (ins := [⟨.float, [3]⟩]) n_atom periodic_distance x lattice

def pairwise_periodic_distance {n_atom : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n_atom, 3]⟩,
    lattice : ⟨.float, []⟩
  begin
    let x := pairwise_offset.apply x;
    let r := Xla.vmap (ins := [⟨.float, [n_atom, 3]⟩]) n_atom periodic_distance_vmapped x lattice;
    return r

noncomputable def pairwise_periodic_distance_def {n_atom : ℕ}
  (x : Fin n_atom → Fin 3 → ℝ) (lattice : ℝ) (i j : Fin n_atom) : ℝ :=
  periodic_distance_def (fun k ↦ x i k - x j k) lattice

theorem pairwise_periodic_distance_eq_def {n_atom : ℕ} :
  Xla.simpleEval (pairwise_periodic_distance (n_atom := n_atom)) = pairwise_periodic_distance_def := by
  ext x L i j
  simp only [List.drop_succ_cons, List.drop_zero, Xla.simpleEval, Curry.map,
    pairwise_periodic_distance, Xla.vmap, List.map_cons, List.map_nil, List.cons_append,
    List.nil_append, List.length_cons, List.length_nil, Nat.reduceAdd, Fin.getElem_fin,
    periodic_distance_vmapped, Fin.zero_eta, Fin.isValue, Fin.mk_one, SSA.Expr.eval, Curry.get,
    SSA.evalType.bind, SSA.Impl.bind, Curry.of, Index.unmap, Curry.curry, Fin.coe_ofNat_eq_mod,
    Nat.zero_mod, List.getElem_cons_zero, Index.map, Function.comp_apply, Index.cons, id_eq,
    Fin.succ_zero_eq_one, Curry.uncurry, Index.append, Curry.arg, Curry.pure, Nat.reduceMod,
    List.getElem_cons_succ, Curry.map₂, Index.single, pairwise_periodic_distance_def]
  have := pairwise_offset_eq_def (n_atom := n_atom)
  simp only [List.map_cons, List.map_nil, Xla.simpleEval, Curry.map, Fin.isValue] at this
  rw [funext_iff] at this
  simp only [Fin.isValue, this]
  have := periodic_distance_eq_def
  simp only [List.drop_succ_cons, List.drop_zero, Xla.simpleEval, Curry.map, Fin.isValue] at this
  rw [funext_iff] at this
  conv at this =>
    intro x
    rw [funext_iff]
  simp [this]
  rfl

def rbf {n_atom n_base n_filter : ℕ} :=
  ssa Xla.XlaOp with
    distance : ⟨.float, [n_atom, n_atom]⟩,
    center : ⟨.float, [n_base]⟩,
    width : ⟨.float, [n_base]⟩,
    filter : ⟨.float, [n_base, n_filter]⟩
  begin
    let distance := Xla.broadcast [⟨n_atom, true⟩, ⟨n_atom, true⟩, ⟨n_base, false⟩] distance;
    let center := Xla.broadcast [⟨n_atom, false⟩, ⟨n_atom, false⟩, ⟨n_base, true⟩] center;
    let width := Xla.broadcast [⟨n_atom, false⟩, ⟨n_atom, false⟩, ⟨n_base, true⟩] width;
    let rbf := Xla.sub distance center;
    let rbf := Xla.div rbf width;
    let rbf := Xla.mul rbf rbf;
    let rbf := Xla.exp (Xla.neg rbf);
    let rbf := Xla.einsum [n_base, n_atom, n_atom, n_filter] [[1,2,0], [0,3]] 1 (rbf.append filter);
    return rbf
    
noncomputable def rbf_def {n_atom n_base n_filter : ℕ}
  (distance : Fin n_atom → Fin n_atom → ℝ)
  (center width : Fin n_base → ℝ)
  (filter : Fin n_base → Fin n_filter → ℝ)
  (i j : Fin n_atom) (f : Fin n_filter) : ℝ :=
  ∑ b, Real.exp ( -((distance i j - center b) / width b) ^ 2) * filter b f

theorem rbf_eq_def {n_atom n_base n_filter : ℕ} :
    Xla.simpleEval (rbf (n_atom := n_atom) (n_base := n_base) (n_filter := n_filter)) = rbf_def := by
  ext r μ γ M
  unfold rbf_def
  conv_rhs =>
    intro i j f; arg 2; intro b
    rw [pow_two]
  apply funext
  intro i
  apply funext
  intro j
  apply funext
  intro f
  simp only [Xla.simpleEval, Curry.map, List.drop_succ_cons, List.drop_zero, rbf, Xla.einsum,
    Xla.bindPrim, List.length_cons, List.length_nil, Nat.reduceAdd, List.map_cons,
    List.get_eq_getElem, Fin.coe_ofNat_eq_mod, Nat.reduceMod, List.getElem_cons_succ,
    List.getElem_cons_zero, Nat.zero_mod, List.map_nil, Fin.getElem_fin, Xla.exp, Xla.neg, Xla.mul,
    Xla.div, Xla.sub, Xla.broadcast, Fin.zero_eta, Fin.isValue, Fin.mk_one, Fin.reduceFinMk,
    SSA.Expr.eval, Curry.get, SSA.evalType.bind, SSA.Impl.bind, Index.single, SSA.SimpleImpl.bind,
    SSA.Tensor.einsum, SSA.Tensor.sumN, SSA.Tensor.sumFirst, SSA.Tensor.einprod, zero_add,
    List.ofFn_succ, Fin.cast_eq_self, Index.map, Function.comp_apply, Curry.of, Index.cons,
    Fin.val_succ, Fin.succ_zero_eq_one, List.ofFn_zero, List.prod_cons, List.prod_nil, mul_one,
    List.tail_cons, Curry.map₂, Index.append, SSA.Tensor.map, SSA.Tensor.map₂, SSA.Tensor.broadcast,
    Curry.arg, Curry.pure, id_eq, List.nil_append]
  trans (∑ b, fun i j f ↦ Real.exp (-((r i j - μ b) / γ b * ((r i j - μ b) / γ b))) * M b f) i j f
  · rfl
  · simp [Finset.sum_apply]

def message_passing {n_atom n_filter n_feat : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n_atom, n_feat]⟩,
    rbf : ⟨.float, [n_atom, n_atom, n_filter]⟩,
    proj₀ : ⟨.float, [n_feat, n_filter]⟩,
    proj₁ : ⟨.float, [n_feat, n_filter]⟩
  begin
    let args := x.append <| proj₀.append <| rbf.append <| proj₁;
    let x := Xla.einsum [n_filter, n_feat, n_atom, n_atom, n_feat] [[2, 1], [1, 0], [2, 3, 0], [4, 0]] 3 args;
    return x

def message_passing_def {n_atom n_filter n_feat : ℕ}
  (x : SSA.Tensor ℝ [n_atom, n_feat])
  (rbf : SSA.Tensor ℝ [n_atom, n_atom, n_filter])
  (proj₀ proj₁ : SSA.Tensor ℝ [n_feat, n_filter]) :
    SSA.Tensor ℝ [n_atom, n_feat] :=
  let y : SSA.Tensor ℝ [n_atom, n_filter] :=
    fun atom filter => ∑ feat, x atom feat * proj₀ feat filter
  let z : SSA.Tensor ℝ [n_atom, n_filter] :=
    fun atom filter => ∑ atom', rbf atom' atom filter * y atom' filter
  fun atom feat => ∑ filter, z atom filter * proj₁ feat filter

theorem Index.single_zero {ι : Type} {m : ι → Type} {i : ι} {x : m i} : Index.single x 0 = x := rfl

theorem message_passing_eq_def {n_atom n_filter n_feat : ℕ} :
    let mp := message_passing (n_atom := n_atom) (n_filter := n_filter) (n_feat := n_feat)
    Xla.simpleEval mp = message_passing_def := by
  ext x rbf proj₀ proj₁
  apply funext; intro atom
  apply funext; intro feat
  simp only [Xla.simpleEval, Curry.map, List.drop_succ_cons, List.drop_zero, message_passing,
    Xla.einsum, Xla.bindPrim, List.length_cons, List.length_nil, Nat.reduceAdd, List.map_cons,
    List.get_eq_getElem, Fin.coe_ofNat_eq_mod, Nat.reduceMod, List.getElem_cons_succ,
    List.getElem_cons_zero, List.map_nil, Nat.zero_mod, Fin.getElem_fin, List.cons_append,
    List.nil_append, Fin.zero_eta, Fin.isValue, Fin.reduceFinMk, Fin.mk_one, SSA.Expr.eval,
    Curry.get, SSA.evalType.bind, SSA.Impl.bind, SSA.SimpleImpl.bind, SSA.Tensor.einsum,
    SSA.Tensor.sumN, SSA.Tensor.sumFirst, zero_add, List.ofFn_succ, Fin.cast_eq_self, Index.map,
    Function.comp_apply, Curry.of, Fin.succ_zero_eq_one, Fin.succ_one_eq_two, Index.cons,
    Fin.val_succ, Fin.reduceSucc, List.ofFn_zero, List.tail_cons, Finset.sum_apply, Curry.map₂,
    Index.append, Curry.arg, Curry.pure, Index.single_zero, message_passing_def]
  conv_lhs =>
    arg 2; intro atom'; arg 2; intro feat'; arg 2; intro filter
    change x atom' feat' * (proj₀ feat' filter * (rbf atom' atom filter * (proj₁ feat filter * 1)))
  conv_rhs =>
    arg 2; intro filter
    rw [Finset.sum_mul]
    arg 2; intro atom'
    rw [Finset.mul_sum, Finset.sum_mul]
  nth_rw 2 [Finset.sum_comm]
  congr
  ext atom'
  nth_rw 1 [Finset.sum_comm]
  ac_nf

def SchNet {n_atom n_species n_base n_filter n_feat : ℕ} :=
  ssa Xla.XlaOp with
    x : ⟨.float, [n_atom, 3]⟩,
    lattice : ⟨.float, []⟩,
    Z : ⟨.int, [n_atom]⟩,
    θ : ⟨.float, [n_species, n_feat]⟩,
    center : ⟨.float, [n_base]⟩,
    width : ⟨.float, [n_base]⟩,
    filter : ⟨.float, [n_base, n_filter]⟩,
    proj₀ : ⟨.float, [n_feat, n_filter]⟩,
    proj₁ : ⟨.float, [n_feat, n_filter]⟩
  begin
    let d := pairwise_periodic_distance.apply (x.append lattice);
    let x := Xla.vmap (ins := [⟨.int, []⟩]) n_atom (embed n_species n_feat) Z θ;
    let w := rbf.apply <| d.append <| center.append <| width.append <| filter;
    let y := message_passing.apply <| x.append <| w.append <| proj₀.append <| proj₁;
    return y

#eval IO.println (SchNet (n_atom := 64) (n_species := 2) (n_base := 16) (n_filter := 8) (n_feat := 16)).code

