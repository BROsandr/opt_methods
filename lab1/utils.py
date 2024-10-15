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

  answer_point = k_points[-1]
  points_x = [point.x for point in k_points[:-1]]
  points_y = [point.y for point in k_points[:-1]]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')
  for i, (xi, yi) in enumerate(zip(points_x, points_y)):
    ax.text(xi, yi, f'{i+1}', fontsize=12, ha='left')

  ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')
  plot_x_eps(ax=ax, origin=star_point, eps=eps)
  ax.scatter(answer_point.x, answer_point.y, c='g', label='$x_n$')

  ax.set(xlabel='x', ylabel='y',
        title='Brute force')

def plot_arrow(ax, origin: Point, f, direction):
  dx = 0.0001 if direction >= 0 else -0.0001
  dy = f(origin.x + dx) - origin.y
  ax.arrow(origin.x, origin.y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=.02)

def plot_bitwise_search(ax: Axes, f: Callable[[Any], Any], a, b, star_point: Point, k_points: list[Point], eps):
  x = np.arange(a, b, 0.001)
  y = [f(y) for y in x]

  ax.plot(x, y)

  points_x = [point.x for point in k_points]
  points_y = [point.y for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')
  for i, (xi, yi) in enumerate(zip(points_x, points_y)):
    ax.text(xi, yi, f'{i+1}', fontsize=12, ha='left')
  for old_point, new_point in zip(k_points[:-1], k_points[1:]):
    if old_point.x < new_point.x:
      plot_arrow(ax=ax, origin=new_point, f=f, direction=1)
    else:
      plot_arrow(ax=ax, origin=new_point, f=f, direction=-1)

  ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')
  plot_x_eps(ax=ax, origin=star_point, eps=eps)

  ax.set(xlabel='x', ylabel='y',
        title='Bitwise search')
