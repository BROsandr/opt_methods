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

def bitwise_search(f: Callable[[Any], Any], a, b, eps, get_init_delta=lambda: 0.25)->Point:
  assert a <= b
  assert eps > 0

  delta1 = get_init_delta()
  assert get_init_delta > eps
  points = {}

  while True:
    delta1_x = np.arange(a, b + delta1, delta1)

    def get_f(x):
      if x in points:
        return points[x]
      else:
        return f(x)

    delta1_y = [get_f(delta1_x[0])]

    endpoint_min = True
    for f_curr, f_next in zip((get_f(xi) for xi in delta1_y + delta1_x[1:-1]),
                              (get_f(xi) for xi in delta1_x[1:])):
      delta1_y.append(f_next)
      if f_curr <= f_next:
        endpoint_min = False
        break

    if endpoint_min:
      return Point(x=delta1_x[-1], y=delta1_y[-1])

    if delta1 <= eps:
      return Point(x=delta1_x[-2], y=delta1_y[-2])

    points.update(dict(zip(delta1_x, delta1_y)))

    delta1 = -delta1 / 4
    a, b = b, a
