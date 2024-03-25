import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LinearRegression:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.b = [0, 0]  # initial values for a0 and a1 formula y = a0 + a1*x

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
    # update data for protect division by zero
    p = p[e != 0]
    e = e[e != 0]
    # Mean Absolute Error
    return 1 - np.mean(np.abs((p - e) / e))

  def plot_best_fit(self, Y_pred, title):
    plt.scatter(self.X, self.Y, color='b')
    plt.plot(self.X, Y_pred, color='g')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
  X = np.array([i for i in range(11)])
  Y = np.array([2*i for i in range(11)])  # y = 2x

  regression = LinearRegression(X, Y)

  # original best-fit line
  Y_pred = regression.predict()
  # regression.plot_best_fit(Y_pred, 'Initial Best Fit Line')
  # print("Initial accuracy is :", regression.get_current_accuracy(Y_pred))

  # Gradient Descent
  epochs = 1000  # number of iterations
  learning_rate = 0.01  # step size

  # splot anamation
  fig, ax = plt.subplots()
  line = ax.plot(X, Y, color='b')
  ax.plot(X, Y, 'ro')

  current_epoch = ax.text(0.02, 0.95, "", transform=ax.transAxes)
  current_epoch.set_text("Epoch: 0")

  current_accuracy = ax.text(0.02, 0.90, "", transform=ax.transAxes)
  current_accuracy.set_text(f"Accuracy: {regression.get_current_accuracy(Y_pred)}")

  current_cost_function_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)
  current_cost_function_text.set_text("Cost Function: ")

  current_function = ax.text(0.02, 0.80, "", transform=ax.transAxes)
  current_function.set_text("Function: y = a0 + a1*x")

  plt.title('Gradient Descent')
  plt.xlabel('X')
  plt.ylabel('Y')

  def animate(i):
    global Y_pred
    Y_pred = regression.predict()
    regression.update_coeffs(learning_rate)
    line[0].set_ydata(Y_pred)
    current_epoch.set_text("Epoch: %d" % i)
    current_accuracy.set_text(f"Accuracy: {regression.get_current_accuracy(Y_pred)}")
    current_cost_function_text.set_text(f"Cost Function: {np.sum((Y_pred - Y) ** 2)}")
    current_function.set_text(f"Function: y = {regression.b[0]} + {regression.b[1]}*x")
    if i == epochs - 1:
      regression.b = [0, 0]  # reset coefficients after reaching the last epoch
    return line

  ani = animation.FuncAnimation(fig, animate, frames=epochs, repeat=True, interval=100)
  plt.show()
