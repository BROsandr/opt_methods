import unittest
import math
from algos import *
from typing import Callable
from utils import Point, LogPointsWrap

def get_y_abs_tol(f: Callable, x, eps):
  assert eps > 0

  y = f(x)

  return max(abs(y - f(x + eps)), abs(y - f(x - eps)))

def f_lecture(x):
  return x**4 + math.exp(-x)

def f_lecture_deriv(x):
  return 4 * x**3 - math.exp(-x)

LECTURE_MIN = Point(x=0.52825, y=0.66750)

def test_lecture_min(test_obj, method: Callable, eps):
  f = f_lecture
  eps = 0.1

  actual_xy = method(f=f, a=0, b=1, eps=eps)

  atol = get_y_abs_tol(f=f, x=LECTURE_MIN.x, eps=eps)

  test_obj.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
  test_obj.assertTrue(math.isclose(a=actual_xy.y, b=LECTURE_MIN.y, abs_tol=atol))

class TestBruteFroce(unittest.TestCase):
  def test_lecture_min(self):
    test_lecture_min(test_obj=self, method=brute_force, eps=0.1)

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
    log_points = LogPointsWrap(f_lecture)
    brute_force(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(expected_points[i].x, actual_points[i].x))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-2))

class TestBitwiseSearch(unittest.TestCase):
  def test_lecture_min(self):
    test_lecture_min(test_obj=self, method=bitwise_search, eps=0.1)

  def test_lecture_all_points(self):
    expected_points = [
      Point(x=0.0000, y=1.000),
      Point(x=0.2500, y=0.783),
      Point(x=0.5000, y=0.669),
      Point(x=0.7500, y=0.789),
      Point(x=0.6875, y=0.726),
      Point(x=0.6250, y=0.688),
      Point(x=0.5625, y=0.670),
      Point(x=0.4375, y=0.682),
    ]

    eps = 0.1
    log_points = LogPointsWrap(f_lecture)
    bitwise_search(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(expected_points[i].x, actual_points[i].x))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-3))

  def test_endpoint(self):
    f = lambda x: -x
    real_xy = Point(x=1, y=-1)
    eps = 0.1

    log_points = LogPointsWrap(f)
    actual_xy = bitwise_search(f=log_points, a=0, b=1, eps=eps)

    self.assertEqual(actual_xy.x, real_xy.x)
    self.assertEqual(actual_xy.y, real_xy.y)

    expected_points = [
      Point(x=0.00, y=-0.00),
      Point(x=0.25, y=-0.25),
      Point(x=0.50, y=-0.50),
      Point(x=0.75, y=-0.75),
      Point(x=1.00, y=-1.00),
    ]

    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertEqual(expected_points[i].x, actual_points[i].x)
        self.assertEqual(expected_points[i].y, actual_points[i].y)

  def test_inf(self):
    f = f_lecture
    eps = 0.1

    actual_xy = bitwise_search(f=f, a=0, b=math.inf, eps=eps)

    atol = get_y_abs_tol(f=f, x=LECTURE_MIN.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy.y, b=LECTURE_MIN.y, abs_tol=atol))


class TestDichotomy(unittest.TestCase):
  def test_lecture_min(self):
    test_lecture_min(test_obj=self, method=dichotomy, eps=0.1)

  @unittest.expectedFailure
  def test_lecture_all_points(self):
    expected_points = [
      Point(x=0.490, y=0.670),
      Point(x=0.510, y=0.688),
      Point(x=0.735, y=0.771),
      Point(x=0.755, y=0.792),
      Point(x=0.613, y=0.683),
      Point(x=0.633, y=0.691),
      Point(x=0.560, y=0.670),
    ]

    eps = 0.1
    log_points = LogPointsWrap(f_lecture)
    dichotomy(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(a=expected_points[i].x, b=actual_points[i].x, abs_tol=1e-3))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-3))

class TestGoldenRatio(unittest.TestCase):
  def test_lecture_min(self):
    test_lecture_min(test_obj=self, method=golden_ratio, eps=0.1)

class TestParabola(unittest.TestCase):
  def test_lecture_min(self):
    f = f_lecture
    eps = 0.025

    actual_xy = parabola(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=LECTURE_MIN.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertIsNone(actual_xy.y)

  def test_lecture_all_points(self):
    expected_points = [
      Point(x=0.2500, y=0.7827),
      Point(x=0.5000, y=0.6690),
      Point(x=0.7500, y=0.7888),
      Point(x=0.4968, y=0.6694),
      Point(x=0.5224, y=0.6676),
      Point(x=0.5248, y=None),
    ]

    eps = 0.025
    log_points = LogPointsWrap(f_lecture)
    min_point = parabola(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points + [min_point]

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)-1):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(expected_points[i].x, actual_points[i].x, abs_tol=1e-4))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-4))

    self.assertTrue(math.isclose(a=actual_points[-1].x, b=expected_points[-1].x, abs_tol=1e-4))
    self.assertIsNone(actual_points[-1].y)

class TestMidpoint(unittest.TestCase):
  def test_lecture_min(self):
    eps = 0.02

    actual_xy = midpoint(f=f_lecture_deriv, a=0, b=1, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertIsNone(actual_xy.y)

  def test_lecture_all_points(self):
    expected_points = [
      Point(x=0.500, y=-0.107),
      Point(x=0.750, y= 1.215),
      Point(x=0.625, y= 0.441),
      Point(x=0.563, y= 0.142),
      Point(x=0.531, y= 0.012),
    ]

    eps = 0.02
    log_points = LogPointsWrap(f_lecture_deriv)
    midpoint(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(a=expected_points[i].x, b=actual_points[i].x, abs_tol=1e-3))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-3))

class TestChord(unittest.TestCase):
  def test_lecture_min(self):
    eps = 0.05

    actual_xy = chord(f=f_lecture_deriv, a=0, b=1, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertIsNone(actual_xy.y)

  def test_lecture_all_points(self):
    expected_points = [
      Point(x=0.216, y=-0.766),
      Point(x=0.352, y=-0.528),
      Point(x=0.435, y=-0.319),
      Point(x=0.480, y=-0.175),
      Point(x=0.504, y=-0.091),
      Point(x=0.516, y=-0.046),
    ]

    eps = 0.05
    log_points = LogPointsWrap(f_lecture_deriv)
    chord(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points) + 7, len(actual_points))

    for i, actual_point in enumerate(actual_points[2::2]):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(a=expected_points[i].x, b=actual_point.x, abs_tol=1e-3))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_point.y, abs_tol=1e-3))

class TestNewton(unittest.TestCase):
  @staticmethod
  def newton_f_d1_lecture(x):
    return math.atan(x)

  @staticmethod
  def newton_f_d2_lecture(x):
    return 1 / (1 + x * x)

  EPS = 1e-7
  X0 = 1

  def test_lecture_min(self):
    eps = self.EPS

    x = newton(f=self.newton_f_d1_lecture, f_deriv=self.newton_f_d2_lecture, x0=self.X0, eps=eps)

    self.assertAlmostEqual(x, LECTURE_MIN.x, delta=eps)

  def test_lecture_all_points(self):
    expected_points = [
      Point(x=1, y=0.785),
      Point(x=-0.570, y=-0.519),
      Point(x=0.117, y=0.116),
      Point(x=-1.061e-3, y=-1.061e-3),
      Point(x=9e-8, y=9e-8),
    ]

    eps = self.EPS
    log_points = LogPointsWrap(self.newton_f_d1_lecture)
    newton(f=log_points, f_deriv=self.newton_f_d2_lecture, x0=self.X0, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertAlmostEqual(expected_points[i].x, actual_points[i].x, places=3)
        self.assertAlmostEqual(expected_points[i].y, actual_points[i].y, places=3)

if __name__ == '__main__':
    unittest.main()
