from functools import partial
import unittest
import math
from algos import *
from utils import *
from typing import Callable
from utils import Point, LogPointsWrap
from plot import *

def should_draw(test_case: unittest.TestCase)->bool:
  return True

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
    eps_point = brute_force(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    if should_draw(self):
      plotting_f = partial(plot_brute_force, f=f_lecture, a=0, b=1, star_point=LECTURE_MIN, eps_point=eps_point, k_points=actual_points, eps=eps, title='Перебор')
      draw_single_plot(plotting_f=plotting_f)

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
    eps_point = bitwise_search(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    if should_draw(self):
      plotting_f = partial(plot_brute_force, f=f_lecture, a=0, b=1, star_point=LECTURE_MIN, eps_point=eps_point, k_points=actual_points, eps=eps, title='Поразрядный поиск')
      draw_single_plot(plotting_f=plotting_f)

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
    eps_point = dichotomy(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    if should_draw(self):
      plotting_f = partial(plot_brute_force, f=f_lecture, a=0, b=1, star_point=LECTURE_MIN, eps_point=eps_point, k_points=actual_points, eps=eps, title='Дихотомия')
      draw_single_plot(plotting_f=plotting_f)

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(a=expected_points[i].x, b=actual_points[i].x, abs_tol=1e-3))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-3))

class TestGoldenRatio(unittest.TestCase):
  def f(self, x):
    return 6 * x**2 + 3*x + 5

  MIN_POINT = Point(x=-1/4, y=37/8)

  def test_parabola(self):
    f = self.f
    log_points = LogPointsWrap(f)
    eps = 0.1
    a, b = -3, 1

    eps_point = golden_ratio(f=log_points, a=-3, b=1, eps=eps)

    atol = get_y_abs_tol(f=log_points, x=self.MIN_POINT.x, eps=eps)

    self.assertTrue(math.isclose(a=eps_point.x, b=self.MIN_POINT.x, abs_tol=eps))
    self.assertTrue(math.isclose(a=eps_point.y, b=self.MIN_POINT.y, abs_tol=atol))

    actual_points = log_points.points

    if should_draw(self):
      plotting_f = partial(plot_brute_force, f=f, a=a, b=b, star_point=self.MIN_POINT, eps_point=eps_point, k_points=actual_points, eps=eps, title='Золотое сечение')
      draw_single_plot(plotting_f=plotting_f)

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

    if should_draw(self):
      plotting_f = partial(plot_parabola_meth, f=f_lecture, a=0, b=1, star_point=LECTURE_MIN, eps_point=min_point, k_points=log_points.points, eps=eps)
      draw_single_plot(plotting_f=plotting_f)

  def test_init_points_gr(self):
    f = f_lecture
    eps = 0.025

    actual_xy = parabola(f=f, a=0, b=1, eps=eps, get_init_points=get_init_points_gr)

    atol = get_y_abs_tol(f=f, x=LECTURE_MIN.x, eps=eps)

    self.assertTrue(math.isclose(a=actual_xy.x, b=LECTURE_MIN.x, abs_tol=eps))
    self.assertIsNone(actual_xy.y)

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
    eps_point = midpoint(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    for i in range(len(expected_points)):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(a=expected_points[i].x, b=actual_points[i].x, abs_tol=1e-3))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_points[i].y, abs_tol=1e-3))

    if should_draw(self):
      plotting_f = partial(plot_midpoint, f=f_lecture, a=0, b=1, star_point=LECTURE_MIN, eps_point=eps_point, k_points=actual_points)
      draw_single_plot(plotting_f=plotting_f)

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
    eps_point = chord(f=log_points, a=0, b=1, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points) + 7, len(actual_points))

    for i, actual_point in enumerate(actual_points[2::2]):
      with self.subTest(i=i):
        self.assertTrue(math.isclose(a=expected_points[i].x, b=actual_point.x, abs_tol=1e-3))
        self.assertTrue(math.isclose(a=expected_points[i].y, b=actual_point.y, abs_tol=1e-3))

    if should_draw(self):
      plotting_f = partial(plot_midpoint, f=f_lecture, a=0, b=1, star_point=LECTURE_MIN, eps_point=eps_point, k_points=actual_points, title='Хорды')
      draw_single_plot(plotting_f=plotting_f)

class TestNewtonRaphson(unittest.TestCase):
  @staticmethod
  def newton_f_d1_lecture(x):
    return math.atan(x)

  @staticmethod
  def newton_f_d2_lecture(x):
    return 1 / (1 + x * x)

  EPS = 1e-7

  def test_newt_lecture_min(self):
    eps = self.EPS
    x0 = 1

    actual_xy = newton(fd1=self.newton_f_d1_lecture, fd2=self.newton_f_d2_lecture, x0=x0, eps=eps)

    self.assertAlmostEqual(actual_xy.x, 0)
    self.assertIsNone(actual_xy.y)

  def test_newt_lecture_all_points(self):
    x0 = 1
    expected_points = [
      Point(x=1, y=0.785),
      Point(x=-0.570, y=-0.519),
      Point(x=0.117, y=0.116),
      Point(x=-1.061e-3, y=-1.061e-3),
      Point(x=9e-8, y=9e-8),
    ]

    eps = self.EPS
    log_points = LogPointsWrap(self.newton_f_d1_lecture)
    newton(fd1=log_points, fd2=self.newton_f_d2_lecture, x0=x0, eps=eps)
    actual_points = log_points.points

    self.assertEqual(len(expected_points), len(actual_points))

    self.assertAlmostEqual(expected_points[0].x, actual_points[0].x, places=3)
    self.assertAlmostEqual(expected_points[0].y, actual_points[0].y, places=3)
    self.assertAlmostEqual(expected_points[1].x, actual_points[1].x, places=2)
    self.assertAlmostEqual(expected_points[1].y, actual_points[1].y, places=3)
    self.assertAlmostEqual(expected_points[2].x, actual_points[2].x, places=3)
    self.assertAlmostEqual(expected_points[2].y, actual_points[2].y, places=3)
    self.assertAlmostEqual(expected_points[3].x, actual_points[3].x, places=5)
    self.assertAlmostEqual(expected_points[3].y, actual_points[3].y, places=5)
    self.assertAlmostEqual(expected_points[4].x, actual_points[4].x, places=6)
    self.assertAlmostEqual(expected_points[4].y, actual_points[4].y, places=6)


  def test_newt_diverge(self):
    eps = self.EPS
    x0 = 3

    with self.assertRaises(ValueError):
      newton(fd1=self.newton_f_d1_lecture, fd2=self.newton_f_d2_lecture, x0=x0, eps=eps)

  def test_newt_raph_lecture_min(self):
    x0 = 3
    eps = self.EPS

    actual_xy = newton(fd1=self.newton_f_d1_lecture, fd2=self.newton_f_d2_lecture, x0=x0, eps=eps, use_tau=True)

    self.assertAlmostEqual(actual_xy.x, 0)
    self.assertIsNone(actual_xy.y)

  def test_marq_lecture_min(self):
    f = lambda x: x * math.atan(x) - 1 / 2 * math.log(1 + x**2)

    x0 = 3
    eps = self.EPS

    actual_xy = newton(fd1=self.newton_f_d1_lecture, fd2=self.newton_f_d2_lecture, x0=x0, eps=eps, f=f)

    self.assertAlmostEqual(actual_xy.x, 0)
    self.assertAlmostEqual(actual_xy.y, 0)

  @staticmethod
  def fd1_lecture2(x):
    return 2 * x - 16 / x**2

  @staticmethod
  def fd2_lecture2(x):
    return 2 + 32 / x**3

  def test_newt_lecture_min2(self):
    eps = self.EPS
    x0 = 1

    actual_xy = newton(fd1=self.fd1_lecture2, fd2=self.fd2_lecture2, x0=x0, eps=eps)

    self.assertAlmostEqual(actual_xy.x, 2)
    self.assertIsNone(actual_xy.y)

  def test_newt_lecture2_all_points(self):
    x0 = 1
    expected_points = [
      Point(x=1.0000, y=None),
      Point(x=1.4118, y=None),
      Point(x=1.8010, y=None),
      Point(x=1.9790, y=None),
      Point(x=1.9998, y=None),
    ]

    eps = self.EPS
    log_points = LogPointsWrap(self.fd1_lecture2)
    newton(fd1=log_points, fd2=self.fd2_lecture2, x0=x0, eps=eps)
    actual_points = log_points.points

    self.assertAlmostEqual(expected_points[0].x, actual_points[0].x, places=4)
    self.assertAlmostEqual(expected_points[1].x, actual_points[1].x, places=4)
    self.assertAlmostEqual(expected_points[2].x, actual_points[2].x, places=3)
    self.assertAlmostEqual(expected_points[3].x, actual_points[3].x, places=4)
    self.assertAlmostEqual(expected_points[4].x, actual_points[4].x, places=4)

if __name__ == '__main__':
    unittest.main()
