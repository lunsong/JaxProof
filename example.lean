import Mathlib.Tactic.NormNum
import JaxProof

open JAX

namespace test_fun

def n : Expr 2 := .arg "n" 0

def x : Expr 2 := .arg "x" 1

def loop_fun_carrier : Expr 4 := .arg "a" 1

def loop_fun : Expr 4 := .mul "b" loop_fun_carrier loop_fun_carrier

def y : Expr 2 := .fori_loop "y" n x loop_fun

end test_fun

#eval IO.println (test_fun.y.code "f")

example (n : ℕ) (x : List ℝ) :
    test_fun.y.eval' (.int [n]) (.float x) = .float (x.map (· ^ (2 ^ n))) := by
  simp[test_fun.y, test_fun.n, test_fun.x, test_fun.loop_fun, test_fun.loop_fun_carrier,
    Expr.eval', curry, Expr.eval]
  induction n with
  | zero => simp
  | succ n ih =>
    simp[ih]
    generalize hx' : (List.map (fun x ↦ x ^ 2 ^ n) x) = x'
    generalize hx'' : (List.map (fun x ↦ x ^ 2 ^ (n + 1)) x) = x''
    simp[Array.mul]
    congr
    refine List.ext_get ?_ ?_
    · simp[← hx', ← hx'']
    · intro m h₁ h₂
      simp[← hx', ← hx'']
      conv_rhs =>
        conv =>
          arg 2
          rw [pow_add, pow_one, mul_two]
        rw [pow_add]

  

