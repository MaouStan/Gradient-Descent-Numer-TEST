  def show_table(self, headers):
      # Clear existing table
      for widget in self.table_frame.winfo_children():
          widget.destroy()

      # Create a scrollable frame for the table
      canvas = tk.Canvas(self.table_frame, height=150)  # Set max height to 150
      canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Use pack instead of grid
      scrollbar = tk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=canvas.yview)
      scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Use pack instead of grid
      canvas.configure(yscrollcommand=scrollbar.set)
      table_frame = tk.Frame(canvas)
      table_frame.pack(expand=True, padx=20, pady=20)  # Add padding to center the table horizontally

      # grid configuration
      for i in range(len(headers)):
          table_frame.grid_columnconfigure(i, weight=1)

      for i in range(len(self.data)):
          table_frame.grid_rowconfigure(i, weight=1)

      # Create table headers
      for j, header in enumerate(headers):
          label = tk.Label(table_frame, text=header, relief=tk.RIDGE, width=10)
          label.grid(row=0, column=j, sticky="NESW")

      # Populate table with data
      for i, row in enumerate(self.data, start=1):
          for j, value in enumerate(row):
              label = tk.Label(table_frame, text=value, relief=tk.RIDGE, width=10)
              label.grid(row=i, column=j, sticky="NESW")

      # Update scroll region
      canvas.create_window((0, 0), window=table_frame, anchor=tk.NW)
      table_frame.update_idletasks()
      canvas.configure(scrollregion=canvas.bbox("all"))