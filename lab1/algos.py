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

  while eps_n > eps:
    if y1 <= y2:
      b = x2
      x2 = x1
      y2 = y1
      x1 = a + b - x2

      y1 = f(x1)
    else:
      a = x1
      x1 = x2
      y1 = y2
      x2 = a + b - x1

      y2 = f(x2)

    eps_n = tau * eps_n

  x_min = (a + b) / 2

  return Point(x=x_min, y=f(x_min))

def get_init_points_gr(f: Callable[[Any], Any], a, b)->list[Point]:
  assert a <= b

  x1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
  x2 = a + (math.sqrt(5) - 1) / 2 * (b - a)

  y1 = f(x1)
  y2 = f(x2)

  ret_points = None

  if y1 <= y2:
    ret_points = [Point(x=a, y=None), Point(x=x1, y=y1), Point(x=x2, y=y2)]
  else:
    ret_points = [Point(x=x1, y=y1), Point(x=x2, y=y2), Point(x=b, y=None)]

  assert ret_points[0].x < ret_points[1].x < ret_points[2].x

  return ret_points

def get_fixed_init_points(*args, **kwargs):
  return (Point(x=0.25, y=None), Point(x=0.5, y=None), Point(x=0.75, y=None))

def parabola(f: Callable[[Any], Any], a, b, eps, get_init_points=get_fixed_init_points)->Point:
  assert a <= b
  assert eps > 0

  init_points = get_init_points(f=f, a=a, b=b)
  for i in range(len(init_points)):
    if init_points[i].y is None:
      init_points[i].y = f(init_points[i].x)

  x1, x2, x3 = init_points[0].x, init_points[1].x, init_points[2].x
  f1, f2, f3 = init_points[0].y, init_points[1].y, init_points[2].y
  assert x1 < x2 < x3
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
        x3 = x_min
        f3 = f_min

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
    else:
      a = x_tilda
      fa = fx

def newton(fd1: Callable[[Any], Any], fd2: Callable[[Any], Any],
           x0: Any,	eps, use_tau=False, f=None, kmax: int=1000) -> Point:
  """
  solves f'(x) = 0 by Newton's method with precision eps
  :param fd1: f'
  :param fd2: f''
  :param x0: starting point
  :param eps: precision wanted
  :return: root Point(x, y) of f'(x) = 0
  """

  assert eps > 0

  x, i = x0, 0
  y = f(x) if f is not None else None
  tau = 1.0
  mu = 0

  while i < kmax:
    yd1 = fd1(x)
    yd2 = fd2(x)

    try:
      if use_tau:
        x_tau = x - yd1 / yd2
        yd1_tau = fd1(x_tau)
        tau = (yd1**2) / (yd1**2 + yd1_tau**2)

      if f is not None and mu == 0:
        mu = yd2 * 10

      x_new = x - tau * yd1 / (yd2 + mu)
    except ZeroDivisionError as e:
      raise ValueError(f"The method doesn't converge after the iteration: №{i} with x0: {x0}, ε: {eps}, τ: {tau}, μ: {mu}") from e

    if f is not None:
      y_new = f(x_new)
      if y_new < y:
        mu /= 2
      else:
        mu *= 2

      y = y_new

    x = x_new
    i += 1

    if abs(yd1) <= eps: break

  if i == kmax:
    raise ValueError(f"The method didn't achieve the specified ε after the max iteration: №{i} with x0: {x0}, ε: {eps}, τ: {tau}, μ: {mu}")

  return Point(x=x, y=y)
