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
