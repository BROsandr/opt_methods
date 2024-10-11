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
    real_xy = Point(x=0.5825, y=0.66750)
    eps = 0.1

    actual_xy = brute_force(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=real_xy.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=real_xy.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy.y, b=real_xy.y, abs_tol=atol))

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

class TestBitwiseSearch(unittest.TestCase):
  @staticmethod
  def f_lecture(x):
    return x**4 + math.exp(-x)

  def test_lecture_min(self):
    f = self.f_lecture
    real_xy = Point(x=0.5825, y=0.66750)
    eps = 0.1

    actual_xy = bitwise_search(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=real_xy.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=real_xy.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy.y, b=real_xy.y, abs_tol=atol))

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
    log_points = LogPointsWrap(self.f_lecture)
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
    f = self.f_lecture
    real_xy = Point(x=0.5825, y=0.66750)
    eps = 0.1

    actual_xy = bitwise_search(f=f, a=0, b=math.inf, eps=eps)

    atol = get_y_abs_tol(f=f, x=real_xy.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=real_xy.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy.y, b=real_xy.y, abs_tol=atol))

def f_lecture(x):
  return x**4 + math.exp(-x)

LECTURE_MIN = Point(x=0.5825, y=0.66750)

class TestDichotomy(unittest.TestCase):
  def test_lecture_min(self):
    f = f_lecture
    eps = 0.1

    actual_xy = dichotomy(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=LECTURE_MIN.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy.y, b=LECTURE_MIN.y, abs_tol=atol))

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
    f = f_lecture
    eps = 0.1

    actual_xy = golden_ratio(f=f, a=0, b=1, eps=eps)

    atol = get_y_abs_tol(f=f, x=LECTURE_MIN.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=actual_xy.y, b=LECTURE_MIN.y, abs_tol=atol))

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

if __name__ == '__main__':
    unittest.main()
