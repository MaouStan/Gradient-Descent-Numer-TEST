import csv
import tkinter as tk
from tkinter import filedialog
from tkinter import font

import numpy as np

# from pages.LinearRegressionGUI import LinearRegressionGUI
from algo.batch import batch_gradient_descent

# demo gui first page has 3 buttons to open new windows
# linear regression, logistic regression, and neural network


class DemoGui:
  def __init__(self, master):
    self.master = master
    master.title("Demo GUI")

    # gui to select data and run linear regression
    self.label = tk.Label(master, text="Select a CSV file:")
    self.label.pack()

    self.file_button = tk.Button(master, text="Open", command=self.open_csv_file)
    self.file_button.pack()

  def open_csv_file(self):
    file_path = tk.filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
      with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')  # Set delimiter to '\t' for tab-separated values
        # copy column headers
        headers = next(csv_reader)[0].split(',')
        self.data = [row[0].split(',') for row in csv_reader]
        self.X = np.array([float(row[0]) for row in self.data], dtype=np.float64)
        self.Y = np.array([float(row[1]) for row in self.data], dtype=np.float64)
        self.show_buttons()

  def show_buttons(self):
    self.batch_gradient_button = tk.Button(self.master, text="Batch Gradient", command=self.open_batch_gradient, fg="white", bg="blue", font=("Arial", 18, font.BOLD))
    self.batch_gradient_button.pack()

  def open_batch_gradient(self):
    # batch_gradient_window = tk.Toplevel(self.master)
    # batch_gradient_window.title("Linear Regression")
    # batch_gradient_window.geometry("800x600")
    batch_gradient_descent(self.X, self.Y, epochs=10, alpha=0.1)


if __name__ == "__main__":
  root = tk.Tk()
  root.geometry("800x600")
  root.resizable(False, False)
  my_gui = DemoGui(root)
  root.mainloop()
