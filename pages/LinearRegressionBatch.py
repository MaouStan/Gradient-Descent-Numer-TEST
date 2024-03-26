import tkinter as tk
from tkinter import filedialog
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LinearRegressionCalculator:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    # y = mx+b
    self.m = 0  # ความชัน
    self.b = 0  # ค่าคงที่

  def update_coeffs(self, learning_rate):
    Y_pred = self.predict()  # predict Y values based on current coefficients
    Y = self.Y  # actual Y values
    m = len(Y)  # number of samples
    # Update the coefficients using gradient descent MSE
    self.m = self.m - learning_rate * (1 / m) * np.sum((Y_pred - Y) * self.X)
    self.b = self.b - learning_rate * (1 / m) * np.sum(Y_pred - Y)

  def predict(self, X=[]):
    Y_pred = np.array([])
    if not X:
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


class LinearRegressionGUI:
  def __init__(self, master):
    self.master = master
    master.title("Linear Regression GUI")

    # gui to select data and run linear regression
    self.label = tk.Label(master, text="Select a CSV file:")
    self.label.pack()

    self.file_button = tk.Button(master, text="Open", command=self.open_csv_file)
    self.file_button.pack()

    # show table of data
    self.table_frame = tk.Frame(master)
    self.table_frame.pack()

  def open_csv_file(self):
    file_path = tk.filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
      with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')  # Set delimiter to '\t' for tab-separated values
        # copy column headers
        headers = next(csv_reader)[0].split(',')
        self.data = [row[0].split(',') for row in csv_reader]

        # set self.data
        self.show_table(headers)
        self.show_graph_button()

  def show_graph_button(self):
    # input epoch and learning rate default values is 1000 and 0.01
    self.epoch_label = tk.Label(self.master, text="Epochs:")
    self.epoch_label.pack()
    self.epoch_entry = tk.Entry(self.master)
    self.epoch_entry.pack()
    self.epoch_entry.insert(0, "1000")

    self.learning_rate_label = tk.Label(self.master, text="Learning Rate:")
    self.learning_rate_label.pack()
    self.learning_rate_entry = tk.Entry(self.master)
    self.learning_rate_entry.pack()
    self.learning_rate_entry.insert(0, "0.01")

    # is checkbox repeat
    self.repeat = tk.IntVar()
    self.repeat_check = tk.Checkbutton(self.master, text="Repeat", variable=self.repeat)
    self.repeat_check.pack()
    self.repeat.set(1)

    # button to start plot
    self.plot_button = tk.Button(self.master, text="Start Plot", command=self.start_plot)
    self.plot_button.pack()

  def show_table(self, headers):
    # Clear existing table
    for widget in self.table_frame.winfo_children():
      widget.destroy()

    # Create a scrollable frame for the table
    canvas = tk.Canvas(self.table_frame, height=150)  # Set max height to 30
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Use pack instead of grid
    scrollbar = tk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Use pack instead of grid
    canvas.configure(yscrollcommand=scrollbar.set)
    table_frame = tk.Frame(canvas)
    table_frame.pack()

    # grid configuration
    for i in range(len(headers)):
      table_frame.grid_columnconfigure(i, weight=1)

    for i in range(len(self.data)):
      table_frame.grid_rowconfigure(i, weight=1)

    # Create table headers
    for j, header in enumerate(headers):
      label = tk.Label(table_frame, text=header, relief=tk.RIDGE, width=10)
      label.grid(row=0, column=j)

    # Populate table with data
    for i, row in enumerate(self.data, start=1):
      for j, value in enumerate(row):
        label = tk.Label(table_frame, text=value, relief=tk.RIDGE, width=10)
        label.grid(row=i, column=j)

    # Update scroll region
    canvas.create_window((0, 0), window=table_frame, anchor=tk.NW)
    table_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

  def start_plot(self):
    # get data
    X = np.array([float(row[0]) for row in self.data])
    Y = np.array([float(row[1]) for row in self.data])

    # get epochs and learning rate
    epochs = int(self.epoch_entry.get())
    learning_rate = float(self.learning_rate_entry.get())

    # is repeat
    repeat = self.repeat.get()

    # Create a LinearRegressionCalculator object
    lr = LinearRegressionCalculator(X, Y)
    lr.plot_animation(epochs=epochs, learning_rate=learning_rate, is_repeat=repeat)


if __name__ == "__main__":
  root = tk.Tk()
  root.geometry("800x600")
  root.resizable(False, False)
  my_gui = LinearRegressionGUI(root)
  root.mainloop()
