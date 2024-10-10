import unittest
import math
from algos import *
from typing import Callable

def get_y_abs_tol(f: Callable, x, eps):
  assert eps > 0

  y = f(x)

  return max(abs(y - f(x + eps)), abs(y - f(x - eps)))

class TestBruteFroce(unittest.TestCase):
  def test_lecture(self):
    def f(x):
      return x**4 + math.exp(-x)

    real_xy = (0.5825, 0.66750)
    eps = 0.1

    actual_xy = brute_force(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=real_xy[0], eps=eps)

    self.assertTrue(math.isclose(a=actual_xy[0], b=real_xy[0], abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy[1], b=real_xy[1], abs_tol=atol))

if __name__ == '__main__':
    unittest.main()
