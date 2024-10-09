import numpy as np

from typing import Callable

def brute_force(f: Callable, a, b, eps):
  assert a <= b
  assert eps != 0

  n = (b - a) / eps

  x = np.arange(a, b + eps, eps)

  return min(
