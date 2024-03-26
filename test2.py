from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LogisticRegressionCalculator:
  def __init__(self, X, y, coefficients=None):
    self.X = X
    self.y = y
    self.w = np.zeros(X.shape[1])  # Coefficients
    self.b = 0.0  # Intercept

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def predict(self, X):
    return self.sigmoid(np.dot(X, self.w) + self.b)

  def update_coeffs(self, learning_rate):
    ypred = self.predict(self.X)
    m = len(self.y)
    grad_w = np.dot(self.X.T, (ypred - self.y)) / m
    grad_b = np.sum(ypred - self.y) / m
    self.w -= learning_rate * grad_w
    self.b -= learning_rate * grad_b

  def plot_animation(self, epochs, learning_rate, is_repeat=True):
    X = self.X
    y = self.y
    y_pred = self.predict(X)
    fig, ax = plt.subplots()
    line = ax.plot(X, y_pred, color='g')  # Note the comma after line
    ax.plot(X, y, 'ro')
    current_epoch = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
      line[0].set_ydata([np.nan] * len(X))
      current_epoch.set_text("")
      return line, current_epoch

    def animate(i):
      self.update_coeffs(learning_rate)
      y_pred = self.predict(X)
      line[0].set_ydata(y_pred)
      current_epoch.set_text(f"Epoch: {i}")
      return line, current_epoch

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=epochs, repeat=is_repeat, interval=0.00001)

    plt.show()


# Example usage:
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)
X = X.squeeze()
log_reg = LogisticRegressionCalculator(X, y)
log_reg.plot_animation(epochs=10000, learning_rate=0.001)
