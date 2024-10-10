import unittest
import math
from algos import *
from typing import Callable
from utils import Point, LogPointsWrap

def get_y_abs_tol(f: Callable, x, eps):
  assert eps > 0

  y = f(x)

  return max(abs(y - f(x + eps)), abs(y - f(x - eps)))

class TestBruteFroce(unittest.TestCase):
  @staticmethod
  def f_lecture(x):
    return x**4 + math.exp(-x)

  def test_lecture_min(self):
    f = self.f_lecture
    real_xy = (0.5825, 0.66750)
    eps = 0.1

    actual_xy = brute_force(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=real_xy[0], eps=eps)

    self.assertTrue(math.isclose(a=actual_xy[0], b=real_xy[0], abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy[1], b=real_xy[1], abs_tol=atol))

  def test_lecture_all_points(self):
    expected_points = [
      Point(x=0.0, y=1.00),
      Point(x=0.1, y=0.90),
      Point(x=0.2, y=0.82),
      Point(x=0.3, y=0.75),
      Point(x=0.4, y=0.70),
      Point(x=0.5, y=0.67),
      Point(x=0.6, y=0.68),
      Point(x=0.7, y=0.74),
      Point(x=0.8, y=0.86),
      Point(x=0.9, y=1.06),
      Point(x=1.0, y=1.37),
    ]

    eps = 0.1
    log_points = LogPointsWrap(self.f_lecture)
    brute_force(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(expected_points[i].x, actual_points[i].x))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-2))

if __name__ == '__main__':
    unittest.main()
