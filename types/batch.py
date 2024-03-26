import csv
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera


# gradient_descent function to reduce cost and find best coeff and intercept
def gradient_descent(x, y):
  # initialization of coeff and y_intercept
  coeff = y_intercept = 0

  # initialization learning rate i.e. alpha
  alpha = 0.01

  # initialization of iterations or epochs
  epochs = 100

  # initialize figure object
  fig = plt.figure(figsize=(13, 6))

  # allocate memory for Camera class i.e. Celluloid
  camera = Camera(fig)

  # iter_lst for storing number of iterations
  iter_lst = []

  # cost_lst for storing cost i.e. M.S.E.
  cost_lst = []
  n = len(x)

  for i in np.arange(epochs):
    # random initialization of coeff and y-intercept
    y_predicted = coeff*x+y_intercept

    # calculating mean square error (MSE)
    cost = (1/n)*np.sum(np.square(y-y_predicted))

    # append the cost & iterations in respective list
    cost_lst.append(cost)
    iter_lst.append(i)

    # creation of subplots to visualize cost & coeff,intercept
    plt.subplot(121)
    plt.scatter(x, y, color='green')
    plt.text(x=np.min(x), y=np.max(y), s="y-intercept: {:1f} & coeff: {:1f}".format(y_intercept, coeff), fontdict={'fontsize': 12})
    plt.xlabel('Independent Variable (x)')
    plt.ylabel('Response Variable (y)')
    plt.plot(x, y_predicted, color='red')
    plt.subplot(122)
    plt.xlabel('Number of epochs (x)')
    plt.ylabel('Cost function (y)')
    plt.plot(iter_lst, cost_lst, color='blue')
    plt.text(x=np.min(x), y=np.max(cost_lst), s="Cost: {:1f}".format(cost), fontdict={'fontsize': 12})

    # capturing snapshot of each & every iteration
    camera.snap()

    # partial derivative of coeff
    coeff_derivative = -(1/n)*(sum(x*(y-y_predicted)))

    # parital derivative of intercept
    y_intercept_derivative = -(1/n)*(sum(y-y_predicted))

    # update coeff and intercept iteratively
    # newweight = oldweight - learning_rate * partialderivatives
    coeff = coeff - alpha*coeff_derivative
    y_intercept = y_intercept - alpha*y_intercept_derivative

  plt.suptitle("BATCHGRADIENT DESCENT & COST FUNCTION PLOT")

  # animate the snapshots
  animate = camera.animate()

  # save
  animate.save('gradient.gif', writer='pillow')

  # plot visualization
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
    gradient_descent(x, y)
