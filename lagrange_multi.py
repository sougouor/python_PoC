#!/bin/python3

"""
Lagrange multiplier optimisation

from:
https://kitchingroup.cheme.cmu.edu/blog/2013/02/03/Using-Lagrange-multipliers-in-optimization/

seek to maximise the function f(x,y) = x+y
constraint x^2 + y^2 =1

function to maximise is an unbounded plane
constraint is a unit circle

want maximum value of circle on the plane
"""


import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def func(X):
  """
  Construct the Lagrane multiplier augmented function

  construct function 
  delta(x,y; lambda) = f(x,y) + lambdag(x,y)
  constraint
  g(x,y) = x^2 + y^2 - 1 = 0

  do not need to change original function
  g(x,y)=0
  constraint is met
  """

  x = X[0]
  y = X[1]
  L = X[2] # multiplier; lambda reserved keyword;
  return x + y + L * (x**2 + x**2 -1)

def dfunc(X):
  """
  Finding the partial derivatives

  minima/maxima of augmented function are located where all 
  of the partial derivatives of the augmented function are equal to zero

  d_delta/d_x = 0
  d_delta/d_y = 0
  d_delta/d_lambda =0

  solving process
  analytically evaluate the partial derivatives
  solve the unconstrainted resulting equations

  another way to numerically approximate the partial derivatives

  """

  d_lambda = np.zeros(len(X))
  h = 1e-3 # step size used in the finite difference
  for i in range(len(X)):
    d_X = np.zeros(len(X))
    d_X[i] =h
    d_lambda[i] = (func(X+d_X) - func(X-d_X)) / (2*h)
  return d_lambda

x = np.linspace(-1.5, 1.5)

[X, Y] = np.meshgrid(x, x)

fig = plt.figure()
# depreciated
#ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, X + Y)

theta = np.linspace(0,2 * np.pi)
R = 1.0
x1 = R * np.cos(theta)
y1 = R * np.sin(theta)

ax.plot(x1, y1, x1+y1, 'r-')
plt.savefig('images/lagrange-1.png')

# maxima
X1 = fsolve(dfunc, [1, 1, 0])
print(X1, func(X1))

# minima
X2 = fsolve(dfunc, [-1, -1, 0])
print(X2, func(X2))

