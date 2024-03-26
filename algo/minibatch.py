import csv
import numpy as np
from celluloid import Camera

import matplotlib.pyplot as plt


# mini_batch_gradient_descent function to reduce cost and find best coeff and intercept
def mini_batch_gradient_descent(x, y, batch_size=4, epochs=10, alpha=0.01):
  # initialization of coeff and y_intercept
  coeff = y_intercept = 0

  # initialize figure object
  fig = plt.figure(2, figsize=(13, 6))

  # allocate memory for Camera class i.e. Celluloid
  camera = Camera(fig)

  # iter_lst for storing number of iterations
  iter_lst = []

  # cost_lst for storing cost i.e. M.S.E.
  cost_lst = []
  n = len(x)

  for i in range(epochs):
    for j in range(0, n, batch_size):
      end = j + batch_size if j + batch_size <= n else n
      batch_x = x[j:end]
      batch_y = y[j:end]

      # predicted value of y
      y_predicted = coeff * batch_x + y_intercept

      # calculating mean square error (MSE)
      cost = np.square(batch_y - y_predicted).mean()

      # append the cost & iterations in respective list
      cost_lst.append(cost)
      iter_lst.append(i*n+j)

      # creation of subplots to visualize cost & coeff,intercept
      plt.subplot(121)
      plt.scatter(x, y, color='green')
      plt.text(x=np.min(x), y=np.max(y), s="y-intercept: {:1f} & coeff: {:1f}".format(y_intercept, coeff), fontdict={'fontsize': 12})
      plt.xlabel('Independent Variable (x)')
      plt.ylabel('Response Variable (y)')
      plt.plot(x, coeff*x + y_intercept, color='red')

      plt.subplot(122)
      plt.xlabel('Number of epochs (x)')
      plt.ylabel('Cost function (y)')
      plt.plot(iter_lst, cost_lst, color='blue')
      plt.text(x=np.min(iter_lst), y=np.max(cost_lst), s="Cost: {:1f}".format(cost), fontdict={'fontsize': 12})

      # capturing snapshot of each & every iteration
      camera.snap()

      # partial derivative of coeff
      coeff_derivative = -(2/batch_size)*(np.sum(batch_x*(batch_y-y_predicted)))

      # parital derivative of intercept
      y_intercept_derivative = -(2/batch_size)*(np.sum(batch_y-y_predicted))

      # update coeff and intercept iteratively
      # newweight = oldweight - learning_rate * partialderivatives
      coeff = coeff-alpha*coeff_derivative
      y_intercept = y_intercept-alpha*y_intercept_derivative

  plt.suptitle("MINI-BATCH GRADIENT DESCENT & COST FUNCTION PLOT")
  animate = camera.animate(repeat=True)
  # animate.save('mini_batch.gif', writer='pillow')
  plt.show()


if __name__ == "__main__":
  headers = []
  data = []
  with open("data/linear_data.csv", 'r') as file:
    csv_reader = csv.reader(file, delimiter='\t')  # Set delimiter to '\t' for tab-separated values
    # copy column headers
    headers = next(csv_reader)[0].split(',')
    data = [row[0].split(',') for row in csv_reader]

    x = np.array([float(row[0]) for row in data])
    y = np.array([float(row[1]) for row in data])

    # function call
    mini_batch_gradient_descent(x, y)
