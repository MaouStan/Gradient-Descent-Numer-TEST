import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LinearRegressionMiniBatch:
  def __init__(self, X, Y, batch_size=10):
    self.X = X
    self.Y = Y
    self.batch_size = batch_size
    # y = mx+b
    self.m = 0  # ความชัน
    self.b = 0  # ค่าคงที่

  def update_coeffs(self, learning_rate):
    Y_pred = self.predict()  # predict Y values based on current coefficients
    Y = self.Y  # actual Y values
    m = len(Y)  # number of samples
    # Update the coefficients using gradient descent MSE
    for i in range(0, m, self.batch_size):
      X_batch = self.X[i:i+self.batch_size]
      Y_batch = self.Y[i:i+self.batch_size]
      Y_pred_batch = self.predict(X_batch)
      self.m = self.m - learning_rate * (1 / self.batch_size) * np.sum((Y_pred_batch - Y_batch) * X_batch)
      self.b = self.b - learning_rate * (1 / self.batch_size) * np.sum(Y_pred_batch - Y_batch)

  def predict(self, X=None):
    Y_pred = np.array([])
    if X is None:
      X = self.X
    b = self.b
    for x in X:
      Y_pred = np.append(Y_pred, b + self.m * x)

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

  def plot_animation(self, epochs, learning_rate, is_repeat=True):
    epochs = epochs + 1
    Y_pred = self.predict()
    # splot anamation
    fig, ax = plt.subplots()
    # line = ax.plot(self.X, self.Y, color='b')
    line = ax.plot(self.X, Y_pred, color='g')
    ax.plot(self.X, self.Y, 'ro')

    current_epoch = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    current_epoch.set_text("Epoch: 0")

    current_accuracy = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    current_accuracy.set_text(f"Accuracy: {self.get_current_accuracy(Y_pred)}")

    current_cost_function_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)
    current_cost_function_text.set_text("Cost Function: ")

    current_function = ax.text(0.02, 0.80, "", transform=ax.transAxes)
    current_function.set_text("Function: y = a0 + a1*x")

    plt.title('Gradient Descent')
    plt.xlabel('X')
    plt.ylabel('Y')

    def animate(i):
      global Y_pred
      Y_pred = self.predict()
      self.update_coeffs(learning_rate)
      line[0].set_ydata(Y_pred)
      current_epoch.set_text("Epoch: %d" % i)
      current_accuracy.set_text(f"Accuracy: {self.get_current_accuracy(Y_pred)}")
      current_cost_function_text.set_text(f"Cost Function: {np.sum((Y_pred - self.Y) ** 2)}")
      current_function.set_text(f"Function: y = {self.b} + {self.m}*x")
      if i == epochs - 1:
        self.b = 0
        self.m = 0
      return line

    ani = animation.FuncAnimation(fig, animate, frames=epochs, repeat=is_repeat, interval=0.1)
    plt.show()


# read csv
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

    # create LinearRegression object
    lr = LinearRegressionMiniBatch(x, y, batch_size=32)
    lr.plot_animation(epochs=1000, learning_rate=0.0001, is_repeat=False)
