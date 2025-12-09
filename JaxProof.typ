#set text(font:"Noto Sans CJK SC")

#show heading: it => {v(5pt); it; v(5pt)}

= JaxProof -- 基于 Lean4 的可验证的科学计算代码生成框架

科学计算中，要验证一个程序的正确性，往往将其同解析解比较，而很多时候没有解析解，这时可以通过形式化方法验证。本项目基于 Lean 这种形式化语言。利用 Lean 灵活的可扩展性，我们可以在 Lean 中*编写函数，验证函数的数学性质，并生成对应的 Python 代码*。下面以一个问题为例，介绍我们期望 AI 的输出 

== 问题描述

假如我们想让 AI 生成一个算法，将一个反对称矩阵相似的变为一个带状矩阵，并要求 AI 证明自己生成代码的正确性。例如，以下算法

```lean
jax_def (shape_of_A : ℕ) houseHolder_for_antisymm(A):
  -- detail goes here
```

然后我们要求 AI 证明，如果输入是反对称矩阵，

1. 输出是一个反对称矩阵（保持原有性质）
2. 结果是三对角的（`i.val + 1 < j.val → B i j = 0`）
3. 存在一个正交矩阵 `C` 使得 `B = C * A * C.transpose`

```lean
example (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
  (h_A_antisymm : ∀ i j, A i j = - A j i) :
    let output := (Jax.native (houseHolder_for_antisymm n)) (Jax.Array.ofMatrix A)
    ∃ B : Matrix (Fin n) (Fin n) ℝ,
      output = Jax.Array.ofMatrix B
      ∧ ∀ i j, B i j = - B j i
      ∧ ∀ i j, i.val + 1 < j.val → B i j = 0
      ∧ ∃ C : Matrix (Fin n) (Fin n) ℝ, B = C * A * C.transpose
      := by
  -- proof of correctness
```
