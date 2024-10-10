import numpy as np

from typing import Callable, Any

import math

def brute_force(f: Callable[[Any], Any], a, b, eps)->tuple[Any, Any]:
  assert a <= b
  assert eps > 0

  n = int(math.ceil((b - a) / eps))

  x = np.linspace(a, b, n + 1, endpoint=True)
  y = np.array([f(el) for el in x])

  ind_min = np.argmin(y)

  return (x[ind_min], y[ind_min])
