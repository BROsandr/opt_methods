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

def plot_x_eps(ax: Axes, origin: Point, eps):
  ax.plot([origin.x - eps, origin.x + eps], [origin.y]*2, 'k|-')

def plot_brute_force(ax: Axes, f: Callable[[Any], Any], a, b, star_point: Point, k_points: list[Point], eps):
  x = np.arange(a, b, 0.001)
  y = [f(y) for y in x]

  ax.plot(x, y)

  points_x = [point.x for point in k_points]
  points_y = [point.y for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')

  ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')

  ax.set(xlabel='x', ylabel='y',
        title='Brute force')
