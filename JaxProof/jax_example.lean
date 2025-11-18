import SecondQuantization.JAX

open JAX

def x : Expr 1 := .arg "x" 1

def loop_fun_carrier : Expr 3 := .arg "a" 1

def loop_fun : Expr 3 := .add "b" loop_fun_carrier loop_fun_carrier

def y : Expr 1 := .fori_loop "y" 3 x loop_fun

#eval IO.println (y.code "f")

example (x : List ℝ) : y.eval' (.float x) = .float (x.map (8 * ·)) := by
  let a := (Array.float x).add (Array.float x)
  let b := a.add a
  let c := b.add b
  change c = .float (x.map (8 * ·))
  simp[c,b,a,Array.add]
  refine List.ext_get ?_ ?_
  · simp
  · intro n h₁ h₂
    simp[←mul_two]
    simp only [mul_assoc]
    nth_rw 1 [mul_comm]
    norm_num
