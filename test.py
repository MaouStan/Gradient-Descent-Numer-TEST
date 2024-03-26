"""
    # slr.py
    This is stochastic logistic regression module
    To use:
        from slr import SLR
        # Import the SLR class from the module and use its methods
        log_reg = SLR()      # Initialization with default params
        log_reg.fit(X, y)    # Fit with train set
        log_reg.predict(X)   # Make predictions with test set
        log_reg.score(X,y)   # Get accuracy score

    Method:
        __init__
        __repr__
        sigmoid
        predict
        predict_proba
        fit
        score
"""
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


class SLR(object):
  """
  This is the SLR class
  """

  def __init__(self, learning_rate=10e-3, n_epochs=10_000, cutoff=0.5):
    """
    The __init__ method
    Params:
        learning_rate
        n_epochs
        cutoff
    """
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.cutoff = cutoff

    self.w = None
    self.b = 0.0

  def __repr__(self):
    params = {
        'learning_rate': self.learning_rate,
        'n_epochs': self.n_epochs,
        'cutoff': self.cutoff
    }
    return "SLR({0}={3}, {1}={4}, {2}={5})".format(*params.keys(), *params.values())

  def sigmoid(self, z):
    """
    The sigmoid method:
    Param:
        z
    Return:
        1.0 / (1.0 + exp(-z))
    """
    return 1.0 / (1.0 + np.exp(-z))

  def predict_proba(self, row):
    """
    The predict_proba
    Param:
        row
    Return:
        sigmoid(z)
    """
    z = np.dot(row, self.w) + self.b
    return self.sigmoid(z)

  def predict(self, X):
    if not isinstance(X, np.ndarray):
      X = X.to_numpy()

    self.predict_probas = []
    for i in range(X.shape[0]):
      ypred = self.predict_proba(X[i])
      self.predict_probas.append(ypred)

    return (np.array(self.predict_probas) >= self.cutoff) * 1.0

  def score(self, X, y):
    """
    The score method
    Param
        X, y
    Return
        accuracy_score(y, ypred)
    """
    ypred = self.predict(X)
    y = y.to_numpy()
    return accuracy_score(y, ypred)

  def fit(self, X, y):
    """
    The fit method implement stochastic gradient descent
    Param
        X, y
    Return
        None
    """
    if not isinstance(X, np.ndarray):
      X = X.to_numpy()

    if not isinstance(y, np.ndarray):
      y = y.to_numpy()

    self.w = np.zeros(X.shape[1])
    self.cost = []

    self.m = X.shape[0]
    self.log_loss = {}
    self.cost = []

    for n_epoch in range(1, self.n_epochs + 1):
      losses = []
      for i in range(self.m):
        yhat = self.predict_proba(X[i])
        grad_b = yhat - y[i]
        grad_w = X[i] * (yhat - y[i])

        self.w -= self.learning_rate * grad_w / self.m
        self.b -= self.learning_rate * grad_b / self.m
        loss = -1/self.m * (y[i] * np.log(yhat) + (1 - y[i]) * np.log(1 - yhat))
        losses.append(loss)

      self.cost.append(sum(losses))


# example usage
if __name__ == "__main__":
  # Generate logistic data
  np.random.seed(0)
  X_logistic = np.random.rand(100, 2) * 10
  coefficients = np.array([1, -1])  # Coefficients for logistic regression
  intercept = -5  # Intercept for logistic regression
  logit = np.dot(X_logistic, coefficients) + intercept
  probabilities = 1 / (1 + np.exp(-logit))
  y_logistic = np.random.binomial(1, probabilities)  # Generate binary labels

  # Create a DataFrame
  logistic_df = pd.DataFrame({'X1': X_logistic[:, 0], 'X2': X_logistic[:, 1], 'y': y_logistic})

  # Save DataFrame to CSV
  logistic_df.to_csv('data/logistic_data.csv', index=False)

  X = logistic_df[['X1', 'X2']]
  y = logistic_df['y']

  log_reg = SLR()
  log_reg.fit(X, y)
  # plot sigmoid
  # scatter data
  plt.scatter(X['X1'], X['X2'], c=y, cmap='viridis')
  x = np.linspace(-10, 10, 100)
  y = log_reg.sigmoid(x)
  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('sigmoid(x)')
  plt.title('Sigmoid Function')
  plt.show()
