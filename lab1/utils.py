from typing import Callable, Any
from dataclasses import dataclass
import numpy as np
from matplotlib.axes import Axes

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

def plot_brute_force(ax: Axes, f: Callable[[Any], Any], a, b, points: list[Point], eps):
  x = np.arange(a, b, 0.001)
  y = [f(y) for y in x]

  ax.plot(x, y)

  points_x = [point.x for point in points]
  points_y = [point.y for point in points]
  ax.plot(points_x, points_y)

  ax.set(xlabel='x', ylabel='y',
        title='Brute force')
  ax.grid()
