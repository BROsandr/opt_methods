import numpy as np
from utils import Point
from typing import Callable, Any

import math

def brute_force(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  n = int(math.ceil((b - a) / eps))

  x = np.linspace(a, b, n + 1, endpoint=True)
  y = np.array([f(el) for el in x])

  ind_min = np.argmin(y)

  return Point(x=x[ind_min], y=y[ind_min])

def bitwise_search(f: Callable[[Any], Any], a, b, eps)->Point:
  assert a <= b
  assert eps > 0

  delta1 = eps * (0.25 / 0.1)

  delta1_x = np.arange(a, b + delta1, delta1)
  delta1_y = [f(delta1_x[0])]

  for f_curr, f_next in zip((f(xi) for xi in delta1_y + delta1_x[1:-1]),
                            (f(xi) for xi in delta1_x[1:])):
    delta1_y.append(f_next)
    if f_curr <= f_next: break

  for i, x in enumerate(delta1_y):
    points = dict(zip(round(delta1_x, delta1_y))
  delta2 = delta1 / 4
