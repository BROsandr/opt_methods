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
  ax.axvspan(origin.x - eps, origin.x + eps, color='lightcoral', alpha=0.3)  # lightcoral is a light red color

def plot_brute_force(ax: Axes, f: Callable[[Any], Any], a, b, star_point: Point, eps_point: Point, k_points: list[Point], eps, title):
  x = np.arange(a, b, 0.001)
  y = [f(y) for y in x]

  ax.plot(x, y)

  answer_point = eps_point
  points_x = [point.x for point in k_points]
  points_y = [point.y for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')
  for i, (xi, yi) in enumerate(zip(points_x, points_y)):
    ax.text(xi, yi, f'{i+1}', fontsize=12, ha='left')

  ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')
  plot_x_eps(ax=ax, origin=star_point, eps=eps)
  ax.scatter(answer_point.x, answer_point.y, c='c', label='$x_ε$')

  ax.set(xlabel='x', ylabel='y',
        title=title)

def plot_arrow(ax, origin: Point, f, direction):
  dx = 0.0001 if direction >= 0 else -0.0001
  dy = f(origin.x + dx) - origin.y
  ax.arrow(origin.x, origin.y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=.02)
