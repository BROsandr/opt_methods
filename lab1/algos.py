import numpy as np
from utils import Point
from typing import Callable, Any
import itertools
import math

def brute_force(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  n = int(math.ceil((b - a) / eps))

  x = np.linspace(a, b, n + 1, endpoint=True)
  y = np.array([f(el) for el in x])

  ind_min = np.argmin(y)

  return Point(x=x[ind_min], y=y[ind_min])

def bitwise_search(f: Callable[[Any], Any], a, b, eps, get_init_delta=lambda: 0.25)->Point:
  assert a <= b
  assert eps > 0

  delta1 = get_init_delta()
  assert delta1 > eps
  points = {}

  while True:
    def get_f(x):
      if x in points:
        return points[x]
      else:
        return f(x)

    f_curr = get_f(a)
    points[a] = f_curr

    x = a + delta1

    while x <= b:
      f_next = get_f(x)
      points[x] = f_next
      if f_curr <= f_next: break
      f_curr = f_next
      x += delta1

    if x > b:
      return Point(x=b, y=f_curr)

    if abs(delta1) <= eps:
      return Point(x=x-delta1, y=f_curr)

    delta1 = -delta1 / 4
    a = x

def dichotomy(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  delta = eps / 5

  while True:
    x1 = (b + a - delta) / 2
    x2 = (b + a + delta) / 2

    y1 = f(x1)
    y2 = f(x2)

    if y1 <= y2:
      b = x2
    else:
      a = x1

    eps_n = (b - a) / 2

    if eps_n <= eps: break

  x_min = (a + b) / 2

  return Point(x=x_min, y=f(x_min))

def golden_ratio(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  x1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
  x2 = a + (math.sqrt(5) - 1) / 2 * (b - a)

  y1 = f(x1)
  y2 = f(x2)

  tau = (math.sqrt(5) - 1) / 2
  eps_n = (b - a) / 2

  while eps_n <= eps:
    if y1 <= y2:
      b = x2
      x2 = x1
      y2 = y1
      x1 = b - tau * (b - a)

      y1 = f(x1)
    else:
      a = x1
      x1 = x2
      y1 = y2
      x2 = b - tau * (b - a)

      y2 = f(x2)

    eps_n = tau * eps_n

  x_min = (a + b) / 2

  return Point(x=x_min, y=f(x_min))

def parabola(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  get_init_points = lambda: (0.25, 0.5, 0.75)

  x1, x2, x3 = get_init_points()
  assert x1 < x2 < x3

  f1, f2, f3 = f(x1), f(x2), f(x3)
  assert f1 >= f2 <= f3

  old_x_min = None

  while True:
    a0 = f1
    a1 = (f2 - f1) / (x2 - x1)
    a2 = 1 / (x3 - x2) * ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1))

    x_min = 1 / 2 * (x1 + x2 - a1 / a2)

    if (old_x_min is not None) and (abs(old_x_min - x_min) <= eps): return Point(x=x_min, y=None)

    f_min = f(x_min)

    if x_min < x2:
      if f_min >= f2:
        x1 = x_min
        f1 = f_min
      else:
        x3 = x2
        f3 = f2
        x2 = x_min
        f2 = f_min

    else:
      if f_min < f2:
        x1 = x2
        f1 = f2
        x2 = x_min
        f2 = f_min
      else:
        x1 = x2
        f1 = f2
        x2 = x_min
        f2 = f_min

    old_x_min = x_min

def midpoint(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  while True:
    x_mid = (a + b) / 2
    f_mid = f(x_mid)

    if abs(f_mid) <= eps:
      return Point(x=x_mid, y=None)

    if f_mid > 0:
      b = x_mid
    else:
      a = x_mid

def chord(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  fa = f(a)
  fb = f(b)

  while True:
    x_tilda = a - fa / (fa - fb) * (a - b)

    fx = f(x_tilda)

    if abs(fx) <= eps:
      return Point(x=x_tilda, y=None)

    if fx > 0:
      b = x_tilda
      fb = fx
      fa = f(a)
    else:
      a = x_tilda
      fa = fx
      fb = f(b)

def newton_raphson(f: Callable[[Any], Any], f_deriv: Callable[[Any], Any], x0: Any,
	eps, tau: float=1.0, kmax: int=1000) -> Any:
  """
  solves f(x) = 0 by Newton's method with precision eps
  :param f: f
  :param f_deriv: f'
  :param x0: starting point
  :param eps: precision wanted
  :return: root of f(x) = 0
  """
  x, i = x0, 0
  auto_tau = True if tau is None else False

  while i < kmax:
    fx = f(x)
    fd = f_deriv(x)

    if auto_tau:
      x_tau = x - fx / fd
      fx_tau = f(x_tau)
      tau = (fx**2) / (fx**2 + fx_tau**2)

    x = x - tau * fx / fd
    i += 1

    if abs(fx) <= eps: break

  return x
