import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk


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

    self.quit_button = tk.Button(master, text="Quit", command=master.quit)
    self.quit_button.pack()

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

    # parameters for linear regression
    self.label = tk.Label(self.master, text="Enter initial coefficients:")
    self.label.pack()
    self.a0_label = tk.Label(self.master, text="a0:")
    self.a0_label.pack()
    self.a0_entry = tk.Entry(self.master)
    self.a0_entry.pack()
    self.a0_entry.insert(0, "1")
    self.a1_label = tk.Label(self.master, text="a1:")
    self.a1_label.pack()
    self.a1_entry = tk.Entry(self.master)
    self.a1_entry.pack()
    self.a1_entry.insert(0, "0.5")

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

    # get initial coefficients
    a0 = float(self.a0_entry.get())
    a1 = float(self.a1_entry.get())
    coe = [a0, a1]

    # get epochs and learning rate
    epochs = int(self.epoch_entry.get())
    learning_rate = float(self.learning_rate_entry.get())

    # is repeat
    repeat = self.repeat.get()

    # Create a LinearRegressionCalculator object
    # lr = LinearRegression(X, Y, coe)
    # lr.plot_animation(epochs=epochs, learning_rate=learning_rate, is_repeat=repeat)
