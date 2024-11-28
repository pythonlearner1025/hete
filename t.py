import numpy as np
import matplotlib.pyplot as plt

def logarithmic_smoothing(x_start, x_end, t, max_steps):
    t = max(1, min(t, max_steps))
    decay = 1 - (np.log(t) / np.log(max_steps))
    value = x_end + (x_start - x_end) * decay
    return value

# Parameters
x_start = 0.2
x_end = 0.05
max_steps = 300

# Generate points for plotting
t_values = np.arange(1, max_steps + 1)
smoothed_values = [logarithmic_smoothing(x_start, x_end, t, max_steps) for t in t_values]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t_values, smoothed_values, 'b-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Timestep')
plt.ylabel('Smoothed Value')
plt.title('Logarithmic Smoothing')

# Add horizontal lines for start and end values
plt.axhline(y=x_start, color='r', linestyle='--', alpha=0.5, label=f'Start value ({x_start})')
plt.axhline(y=x_end, color='g', linestyle='--', alpha=0.5, label=f'End value ({x_end})')

plt.legend()
plt.show()