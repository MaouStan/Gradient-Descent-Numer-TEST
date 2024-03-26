import subprocess

commands = [
    "python algo/batch.py",
    "python algo/minibatch.py",
    "python algo/stochastic.py"
]

# List to hold process objects for each command
processes = []

# Start each command as a separate process
for cmd in commands:
  # Run the command in the background without waiting
  process = subprocess.Popen(cmd, shell=True)
  processes.append(process)

# Wait for all processes to finish
for process in processes:
  process.wait()
