import numpy as np

from typing import Callable, Any

def brute_force(f: Callable[[Any], Any], a, b, eps)->tuple[Any, Any]:
  assert a <= b
  assert eps >= 0

  n = (b - a) / eps

  x = np.linspace(a, b, n, endpoint=True)
  y = f(x)

  ind_min = np.argmin(y)

  return (x[ind_min], y[ind_min])
