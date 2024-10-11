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
    delta1_y = [f_curr]

    x = a + delta1

    while x <= b:
      f_next = get_f(x)
      points[x] = f_next
      delta1_y.append(f_next)
      if f_curr <= f_next: break
      f_curr = f_next
      x += delta1

    if abs(delta1) <= eps:
      return Point(x=x-delta1, y=f_curr)

    delta1 = -delta1 / 4
    a = x
