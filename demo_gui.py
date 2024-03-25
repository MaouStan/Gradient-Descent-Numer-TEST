import tkinter as tk
import tkinter as tk
from tkinter import font

from pages.LinearRegressionGUI import LinearRegressionGUI

# demo gui first page has 3 buttons to open new windows
# linear regression, logistic regression, and neural network


class DemoGui:
  def __init__(self, master):
    self.master = master
    master.title("Demo GUI")

    # layout grid
    for i in range(3):
      master.grid_rowconfigure(i, weight=1)
      master.grid_columnconfigure(0, weight=1)

    self.linear_regression_button = tk.Button(master, text="Linear Regression", command=self.open_linear_regression, fg="white", bg="blue", font=("Arial", 18, font.BOLD))
    self.linear_regression_button.grid(row=0, column=0, sticky="ew", padx=10, pady=30)

    self.logistic_regression_button = tk.Button(master, text="Logistic Regression", command=self.open_logistic_regression, fg="white", bg="green", font=("Arial", 18, font.BOLD))
    self.logistic_regression_button.grid(row=1, column=0, sticky="ew", padx=10, pady=30)

    self.neural_network_button = tk.Button(master, text="Neural Network", command=self.open_neural_network, fg="white", bg="orange", font=("Arial", 18, font.BOLD))
    self.neural_network_button.grid(row=2, column=0, sticky="ew", padx=10, pady=30)

  def open_linear_regression(self):
    linear_regression_window = tk.Toplevel(self.master)
    linear_regression_window.title("Linear Regression")
    linear_regression_window.geometry("800x600")
    LinearRegressionGUI(linear_regression_window)

  def open_logistic_regression(self):
    ...

  def open_neural_network(self):
    ...


if __name__ == "__main__":
  root = tk.Tk()
  root.geometry("800x600")
  root.resizable(False, False)
  my_gui = DemoGui(root)
  root.mainloop()
