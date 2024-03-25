import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
  def __init__(self, X, Y, B=[]):
    self.X = X
    self.Y = Y
    # initial values for a0 and a1 formula y = a0 + a1*x
    if not B:
      self.b = [0, 0]
    else:
      self.b = B

  def update_coeffs(self, learning_rate):
    Y_pred = self.predict()  # predict Y values based on current coefficients
    Y = self.Y  # actual Y values
    m = len(Y)  # number of samples
    # Update the coefficients using gradient descent
    self.b[0] = self.b[0] - (learning_rate * (1/m) * np.sum(Y_pred - Y))  # update a0
    self.b[1] = self.b[1] - (learning_rate * (1/m) * np.sum((Y_pred - Y) * self.X))  # update a1

  def predict(self, X=[]):
    Y_pred = np.array([])
    if not X:
      X = self.X
    b = self.b
    for x in X:
      Y_pred = np.append(Y_pred, b[0] + (b[1] * x))

    return Y_pred

  def get_current_accuracy(self, Y_pred):
    p, e = Y_pred, self.Y
    n = len(Y_pred)
    # Mean Absolute Error
    return 1-sum(
        [
            abs(p[i]-e[i])/e[i]  # relative error
            for i in range(n)  # sum of all relative errors
            if e[i] != 0  # exclude division by zero
        ]
    )/n

  def plot_best_fit(self, Y_pred, title):
    plt.scatter(self.X, self.Y, color='b')
    plt.plot(self.X, Y_pred, color='g')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
  X = np.array([1, 2], dtype=np.float64)
  Y = np.array(X * 2 + 2, dtype=np.float64)

  regression = LinearRegression(X, Y, [1, 0.5])

  # original best-fit line
  Y_pred = regression.predict()
  regression.plot_best_fit(Y_pred, 'Initial Best Fit Line')
  print("Initial accuracy is :", regression.get_current_accuracy(Y_pred))

  # leaning rate
  learning_rate = 0.01
  # number of iterations
  i = 1
  print(Y, Y_pred)
  while regression.get_current_accuracy(Y_pred) < 0.9999:
    regression.update_coeffs(learning_rate)
    Y_pred = regression.predict()
    i += 1

  # print the number of iterations
  print("Number of iterations is :", i)
  # print the final accuracy
  print("Final accuracy is :", regression.get_current_accuracy(Y_pred))
  # print the final coefficients
  print("Final coefficients are :", regression.b)
  # plot the best fit line
  regression.plot_best_fit(Y_pred, 'Best Fit Line')
