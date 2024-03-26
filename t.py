import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(42)

# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = -5 * X
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 0.4 * torch.randn(X.size())

w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

# defining the function for forward pass for prediction


def forward(x):
  return w * x + b

# evaluating data points with Mean Square Error (MSE)


def criterion(y_pred, y):
  return torch.mean((y_pred - y) ** 2)

# Creating our dataset class


class Build_Data(Dataset):
  # Constructor
  def __init__(self):
    self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
    self.y = -5 * X
    self.len = self.x.shape[0]
  # Getting the data

  def __getitem__(self, index):
    return self.x[index], self.y[index]
  # Getting length of the data

  def __len__(self):
    return self.len


# Creating DataLoader object
dataset = Build_Data()
train_loader = DataLoader(dataset=dataset, batch_size=1)

step_size = 0.1
loss_SGD = []
n_iter = 20

for i in range(n_iter):
  # calculating loss as in the beginning of an epoch and storing it
  y_pred = forward(X)
  loss_SGD.append(criterion(y_pred, Y).tolist())
  for x, y in train_loader:
    # making a prediction in forward pass
    y_hat = forward(x)
    # calculating the loss between original and predicted data points
    loss = criterion(y_hat, y)
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    b.grad.data.zero_()

train_loader_10 = DataLoader(dataset=dataset, batch_size=10)

# Reset w and b
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

loss_MBGD_10 = []

for i in range(n_iter):
  # calculating loss as in the beginning of an epoch and storing it
  y_pred = forward(X)
  loss_MBGD_10.append(criterion(y_pred, Y).tolist())
  for x, y in train_loader_10:
    # making a prediction in forward pass
    y_hat = forward(x)
    # calculating the loss between original and predicted data points
    loss = criterion(y_hat, y)
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    b.grad.data.zero_()

train_loader_20 = DataLoader(dataset=dataset, batch_size=20)

# Reset w and b
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)

loss_MBGD_20 = []

for i in range(n_iter):
  # calculating loss as in the beginning of an epoch and storing it
  y_pred = forward(X)
  loss_MBGD_20.append(criterion(y_pred, Y).tolist())
  for x, y in train_loader_20:
    # making a prediction in forward pass
    y_hat = forward(x)
    # calculating the loss between original and predicted data points
    loss = criterion(y_hat, y)
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    w.data = w.data - step_size * w.grad.data
    b.data = b.data - step_size * b.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    b.grad.data.zero_()


# Batch Gradient Descent
loss_BGD = []
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)
for i in range(n_iter):
  # calculating loss as in the beginning of an epoch and storing it
  y_pred = forward(X)
  loss_BGD.append(criterion(y_pred, Y).tolist())
  # making a prediction in forward pass
  y_hat = forward(X)
  # calculating the loss between original and predicted data points
  loss = criterion(y_hat, Y)
  # backward pass for computing the gradients of the loss w.r.t to learnable parameters
  loss.backward()
  # updating the parameters after each iteration
  with torch.no_grad():
    w -= step_size * w.grad
    b -= step_size * b.grad
  # zeroing gradients after each iteration
  w.grad.zero_()
  b.grad.zero_()

# Plotting the loss for each type of gradient descent
plt.plot(loss_BGD, label="Batch Gradient Descent")
plt.plot(loss_SGD, label="Stochastic Gradient Descent")
plt.plot(loss_MBGD_10, label="Mini-Batch-10 Gradient Descent")
plt.plot(loss_MBGD_20, label="Mini-Batch-20 Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/total loss')
plt.legend()
plt.show()
