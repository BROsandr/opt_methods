import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from utils import Point
from typing import Callable, Any

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

def plot_parabola(ax: Axes, points: list[Point]):
  points = sorted(points)

  a0 = points[0].y
  a1 = (points[1].y - points[0].y) / (points[1].x - points[0].x)
  a2 = 1 / (points[2].x - points[1].x) * \
      ((points[2].y - points[0].y) / (points[2].x - points[0].x) - \
       (points[1].y - points[0].y) / (points[1].x - points[0].x))

  f = lambda x: a0 + a1 * (x - points[0].x) + a2 * (x - points[0].x) * (x - points[1].x)

  wings = (points[2].x - points[0].x) / 10
  x = np.arange(points[0].x-wings, points[2].x+wings, 0.001)
  y = [f(y) for y in x]

  ax.plot(x, y, color='orange')

def plot_parabola_meth(ax: Axes, f: Callable[[Any], Any], a, b, star_point: Point, eps_point: Point, k_points: list[Point], eps):
  title = 'Парабола'
  plot_brute_force(ax=ax, f=f, a=a, b=b, star_point=star_point, eps_point=eps_point, k_points=k_points, eps=eps, title=title)

  parabola_points = k_points[:3]
  plot_parabola(ax=ax, points=parabola_points)

  for point in k_points[3:]:
    x_min = point.x
    f_min = point.y
    if x_min < parabola_points[1].x:
      if f_min >= parabola_points[1].y:
        parabola_points[0].x = x_min
        parabola_points[0].y = f_min
      else:
        parabola_points[2].x = parabola_points[1].x
        parabola_points[2].y = parabola_points[1].y
        parabola_points[1].x = x_min
        parabola_points[1].y = f_min

    else:
      if f_min < parabola_points[1].y:
        parabola_points[0].x = parabola_points[1].x
        parabola_points[0].y = parabola_points[1].y
        parabola_points[1].x = x_min
        parabola_points[1].y = f_min
      else:
        parabola_points[2].x = x_min
        parabola_points[2].y = f_min

    plot_parabola(ax=ax, points=parabola_points)

def plot_arrow(ax, origin: Point, f, direction):
  dx = 0.0001 if direction >= 0 else -0.0001
  dy = f(origin.x + dx) - origin.y
  ax.arrow(origin.x, origin.y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=.02)

def draw_single_plot(plotting_f: Callable):
  fig, ax = plt.subplots()
  plotting_f(ax=ax)
  ax.legend()
  ax.grid()
  plt.show()

def plot_tangent(ax: Axes, point: Point, slope, a, b):
  tangent_line = lambda x: point.y + slope * (x - point.x)

  x = np.arange(a, b, 0.001)
  y = [tangent_line(xi) for xi in x]

  ax.plot(x, y)


def plot_midpoint(ax: Axes, f: Callable[[Any], Any], a, b, star_point: Point, eps_point: Point, k_points: list[Point], eps):
  x = np.arange(a, b, 0.001)
  y = [f(xi) for xi in x]

  ax.plot(x, y)

  answer_point = eps_point
  points_x = [point.x for point in k_points]
  points_y = [f(point.x) for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')
  wings = (b - a) / 10
  for i, (point, yi) in enumerate(zip(k_points, points_y)):
    ax.text(point.x, yi, f'{i+1}', fontsize=12, ha='left')
    plot_tangent(ax=ax, point=Point(x=point.x, y=yi), slope=point.y, a=point.x-wings, b=point.x+wings)

  ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')
  ax.scatter(answer_point.x, answer_point.y, c='c', label='$x_ε$')

  ax.set(xlabel='x', ylabel='y',
        title='Средняя точка')
