import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from utils import Point
from typing import Callable, Any
from copy import deepcopy

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

  plot_x_eps(ax=ax, origin=star_point, eps=eps)
  if star_point.x == answer_point.x:
    ax.scatter(star_point.x, star_point.y, c='r', label='$x^*, x_ε$')
  else:
    ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')
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
  k_points_copy = deepcopy(k_points)
  eps_point_copy = deepcopy(eps_point)
  eps_point_copy.y = eps_point_copy.y if eps_point_copy.y is not None else f(eps_point_copy.x)
  plot_brute_force(ax=ax, f=f, a=a, b=b, star_point=star_point, eps_point=eps_point_copy, k_points=k_points_copy, eps=eps, title=title)

  parabola_points = k_points_copy[:3]
  plot_parabola(ax=ax, points=parabola_points)

  for point in k_points_copy[3:]:
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

def draw_double_plot(plotting_f_left: Callable, plotting_f_right: Callable):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  plotting_f_left(ax=ax1)
  plotting_f_right(ax=ax2)
  ax1.legend()
  ax1.grid()
  ax2.legend()
  ax2.grid()
  plt.show()

def plot_tangent(ax: Axes, point: Point, slope, a, b):
  tangent_line = lambda x: point.y + slope * (x - point.x)

  x = np.arange(a, b, 0.001)
  y = [tangent_line(xi) for xi in x]

  ax.plot(x, y)


def plot_midpoint(ax: Axes, f: Callable[[Any], Any], a, b, star_point: Point, eps_point: Point, k_points: list[Point], title='Средняя точка'):
  x = np.arange(a, b, 0.001)
  y = [f(xi) for xi in x]

  ax.plot(x, y)

  answer_point = deepcopy(eps_point)
  points_x = [point.x for point in k_points]
  points_y = [f(point.x) for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')
  wings = (b - a) / 10
  for i, (point, yi) in enumerate(zip(k_points, points_y)):
    ax.text(point.x, yi, f'{i+1}', fontsize=12, ha='left')
    plot_tangent(ax=ax, point=Point(x=point.x, y=yi), slope=point.y, a=point.x-wings, b=point.x+wings)

  answer_point.y = answer_point.y if answer_point.y is not None else f(answer_point.x)

  if star_point.x == answer_point.x:
    ax.scatter(answer_point.x, answer_point.y, c='r', label='$x^*, x_ε$')
  else:
    ax.scatter(star_point.x, star_point.y, c='r', label='$x^*$')
    ax.scatter(answer_point.x, answer_point.y, c='c', label='$x_ε$')

  ax.set(xlabel='x', ylabel='y',
        title=title)

def plot_chord(ax: Axes, fd1: Callable[[Any], Any], a, b, star_point: Point, eps_point: Point, k_points: list[Point]):
  x = np.arange(a, b, 0.001)
  y = [fd1(xi) for xi in x]

  ax.plot(x, y)

  answer_point = deepcopy(eps_point)
  points_x = [point.x for point in k_points]
  points_y = [fd1(point.x) for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')

  ax.text(k_points[0].x, k_points[0].y, f'1', fontsize=12, ha='left')
  ax.text(k_points[1].x, k_points[1].y, f'2', fontsize=12, ha='left')

  for i, point in enumerate(k_points[2:], start=2):
    ax.plot([a, b], [fd1(a), fd1(b)])
    ax.text(point.x, point.y, f'{i+1}', fontsize=12, ha='left')
    ax.plot([point.x, point.x], [0, point.y], linestyle='dashed')

    if point.y > 0:
      b = point.x
    else:
      a = point.x

  if star_point.x == answer_point.x:
    ax.scatter(star_point.x, fd1(star_point.x), c='r', label='$x^*, x_ε$')
  else:
    ax.scatter(star_point.x, fd1(star_point.x), c='r', label='$x^*$')
    ax.scatter(answer_point.x, fd1(answer_point.x), c='c', label='$x_ε$')

  ax.set(xlabel='x', ylabel='y',
        title="f', Хорды")

def plot_newton(ax: Axes,
                fd1: Callable[[Any], Any],
                fd2: Callable[[Any], Any],
                x0,
                star_point: Point,
                eps_point: Point,
                k_points: list[Point]):
  sorted_k_x_points = sorted(map(lambda point: point.x, k_points))
  wings = (max(sorted_k_x_points) - min(sorted_k_x_points)) / 10
  x = np.arange(min(sorted_k_x_points), max(sorted_k_x_points), 0.001)
  y = [fd1(xi) for xi in x]

  ax.plot(x, y)

  answer_point = deepcopy(eps_point)
  points_x = [point.x for point in k_points]
  points_y = [fd1(point.x) for point in k_points]
  ax.scatter(points_x, points_y, c='b', label='$x_k$')

  for i, point in enumerate(k_points):
    ax.text(point.x, point.y, f'{i+1}', fontsize=12, ha='left')

  for i, point in enumerate(k_points[:-1]):
    ax.text(point.x, point.y, f'{i+1}', fontsize=12, ha='left')
    a = point.x - wings if point.y < 0 else k_points[i+1].x - wings
    b = point.x + wings if point.y > 0 else k_points[i+1].x + wings
    plot_tangent(ax=ax, point=Point(x=point.x, y=point.y), slope=fd2(point.x), a=a, b=b)

  for point in k_points[1:]:
    ax.plot([point.x, point.x], [0, point.y], linestyle='dashed')

  if answer_point.x == star_point.x:
    ax.scatter(star_point.x, fd1(star_point.x), c='r', label='$x^*, x_ε$')
  else:
    ax.scatter(answer_point.x, fd1(answer_point.x), c='c', label='$x_ε$')
    ax.scatter(star_point.x, fd1(star_point.x), c='r', label='$x^*$')

  ax.scatter(x0, fd1(x0), c='y', label='$x_0$')

  ax.set(xlabel='x', ylabel='y',
        title="f', Ньютон")
