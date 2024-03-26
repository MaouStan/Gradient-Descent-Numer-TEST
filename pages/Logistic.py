import numpy as np
import csv

import matplotlib.pyplot as plt


class LogisticRegression:
  def __init__(self, learning_rate=0.01, num_iterations=1000):
    self.learning_rate = learning_rate
    self.num_iterations = num_iterations
    self.weights = None
    self.bias = None
    self.costs = []  # Add a list to store the costs

  def _cross_entropy_loss(self, y_predicted, y):
    epsilon = 1e-5  # small value to avoid division by zero
    loss = -np.mean(y * np.log(y_predicted + epsilon) + (1 - y) * np.log(1 - y_predicted + epsilon))
    return loss

  def fit(self, X, y):
    num_samples, num_features = X.shape
    self.weights = np.zeros(num_features)
    self.bias = 0

    for _ in range(self.num_iterations):
      linear_model = np.dot(X, self.weights) + self.bias
      y_predicted = self._sigmoid(linear_model)

      dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
      db = (1 / num_samples) * np.sum(y_predicted - y)

      self.weights -= self.learning_rate * dw
      self.bias -= self.learning_rate * db

      # Calculate and store the cost
      cost = self._cross_entropy_loss(y_predicted, y)
      self.costs.append(cost)

  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_predicted = self._sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_cls

  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
  file_path = "data/logistic_data.csv"
  headers = []
  data = []
  with open(file_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter='\t')  # Set delimiter to '\t' for tab-separated values
    # copy column headers
    headers = next(csv_reader)[0].split(',')
    data = [row[0].split(',') for row in csv_reader]

  # Convert data to numpy arrays
  X1_train = np.array([float(row[0]) for i, row in enumerate(data) if i < 50])
  X2_train = np.array([float(row[1]) for i, row in enumerate(data) if i < 50])
  y_train = np.array([float(row[2]) for i, row in enumerate(data) if i < 50])

  # Create and train the logistic regression model
  X = np.array([X1_train, X2_train]).T
  y = y_train
  model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
  model.fit(X, y)

  # Predict the labels
  X1_test = np.array([float(row[0]) for i, row in enumerate(data) if i >= 50])
  X2_test = np.array([float(row[1]) for i, row in enumerate(data) if i >= 50])
  y_test = np.array([float(row[2]) for i, row in enumerate(data) if i >= 50])

  X_test = np.array([X1_test, X2_test]).T
  y_pred = model.predict(X_test)

  # graph logistic regression curve and data points x,y line costs

  # Calculate the accuracy
  accuracy = np.mean(y_pred == y_test)
  print(f"Accuracy: {accuracy}")
