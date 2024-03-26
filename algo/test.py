import csv

import numpy as np

from stochastic import stochastic_gradient_descent
from batch import batch_gradient_descent
from minibatch import mini_batch_gradient_descent

if __name__ == "__main__":
  headers = []
  data = []
  with open("data/linear_data_10.csv", 'r') as file:
    csv_reader = csv.reader(file, delimiter='\t')  # Set delimiter to '\t' for tab-separated values
    # copy column headers
    headers = next(csv_reader)[0].split(',')
    data = [row[0].split(',') for row in csv_reader]

    x = np.array([float(row[0]) for row in data])
    y = np.array([float(row[1]) for row in data])

    # function call
    batch_gradient_descent(x, y, repeat=False)
    mini_batch_gradient_descent(x, y, repeat=False)
    stochastic_gradient_descent(x, y, repeat=False)
