#!/bin/python3

"""
Gradient Ascent

from:
https://teddykoker.com/2019/05/trading-with-reinforcement-learning-in-python-part-i-gradient-ascent/ 

algorithm used to maximise a given reward function

e.g. 
task to find highet point of the mountain
reward function equals trying to maximise elevation
important to know the slope/gradient to identify direction

learning rate
aka number of steps taken before checking slope again
high learning rate -- algorithm diverge from the maximum
low learning rte algorithm take too long to finish
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import linregress

from mpl_toolkits.mplot3d import Axes3D

# negative MSE function
def accuracy(x, y, m, theta):
  return - 1 / m * np.sum((np.dot(x, theta) - y) ** 2)

# gradient function
def gradient(x, y, m, theta):
  return - 1 /m * x.T.dot(np.dot(x, theta) - y)

# training function
def train(x, y, m, num_epochs):
  accs = []
  thetas = []
  theta = np.zeros(2)
  for _ in range(num_epochs):
    # keep track of accuracy and theta over time
    acc = accuracy(x, y, m, theta)
    thetas.append(theta)
    accs.append(acc)

    # update theta
    theta = theta + learning_rate * gradient(x, y, m, theta)

  return theta, thetas, accs

# generate 100 points placed randomly around a line
# intercept of 5
# slope of 2

plt.rcParams["figure.figsize"] = (5, 3) # (w, h)
plt.rcParams["figure.dpi"] = 200

m = 100
x = 2 * np.random.rand(m)
y = 5 + 2 * x + np.random.randn(m)
plt.scatter(x, y)
plt.show()

# clean plot and axes
plt.clf()
#ax.cla()

# find line of best fits using scipy linregress function
slope, intercept = linregress(x, y)[:2]
print(f"slope: {slope:.3f}, intercept: {intercept:.3f}")

# reward function (J)
# mean square error -- measure accuracy of linear regression
# avg sqr diff between est values and what is est
# 
# use negative MSE for function to maximise

# matrix x where x[0] = 1 x[1] are original values of x 
# theta[0] + theta[1] * x <==> theta * x
x = np.array([np.ones(m), x]).transpose()

# gradient function 
# partial derivative of reward function (J) with respect to theta

# training 
# perform gradient ascent
# initialise theta as [0, 0]
# update every epoch/step
# theta = theta + alpha * gradient function
# alpha learning rate

num_epochs = 500
learning_rate = 0.1

theta, thetas, accs = train(x, y, m, num_epochs)
print(f"slope: {theta[1]:.3f}, intercept: {theta[0]:.3f}")

# match linear benchmark?

plt.plot(accs)
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.show()

# clean plot and axes
plt.clf()
#ax.cla()

# porject reward function onto a 3d surface
# mark theta[0] and theta[1] over time
# observe gradient ascent algorithm gradually finding maximum

i = np.linspace(-10, 20, 50)
j = np.linspace(-10, 20, 50)
i, j = np.meshgrid(i, j)
k = np.array([accuracy(x, y, m, th) for th in zip(np.ravel(i), np.ravel(j))]).reshape(i.shape)
fig = plt.figure(figsize=(9,6))
#ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
ax.plot_surface(i, j, k, alpha=0.2)
ax.plot([t[0] for t in thetas], [t[1] for t in thetas], accs, marker="o", markersize=3, alpha=0.1)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel("Accuracy")

plt.show()

# clean plot and axes
#plt.clf()
#ax.cla()

