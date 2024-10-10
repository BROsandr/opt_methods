from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class Point:
  x: Any
  y: Any

class LogPointsWrap:
  def __init__(self, f:Callable[[Any], Any]):
    self.points = []
    self._f = f

  def __call__(self, x):
    y = self._f(x)
    self.points.append(Point(x=x, y=y))

    return y
