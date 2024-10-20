import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.widgets import Slider, Button

# Generate x values (normally distributed)
x = np.linspace(0, 1, 1000)
y = norm.pdf(x, loc=0.5, scale=0.15)

# Initial p value
init_p = 1.0

# Create the figure and subplot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Plot the normal distribution
line_norm, = ax.plot(x, y, 'b-', label='Normal Distribution')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density / f(x)')

# Plot initial f(x)
f_x = x * (1 - x) #** init_p
line_fx, = ax.plot(x, f_x, 'r-', label='f(x) = (1-x)^p')

# Set the title and display legends
ax.set_title(f'Normal Distribution and f(x) = (1-x)^{init_p:.2f}')
ax.legend(loc='upper right')

# Create a slider for p
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
p_slider = Slider(ax=ax_slider, label='p', valmin=0.1, valmax=10, valinit=init_p, valstep=0.1)

# Update function for the slider
def update(val):
    p = p_slider.val
    f_x = (1 - x) ** p
    line_fx.set_ydata(f_x)
    ax.set_title(f'Normal Distribution and f(x) = (1-x)^{p:.2f}')
    fig.canvas.draw_idle()

# Register the update function with the slider
p_slider.on_changed(update)

# Create a reset button
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')

# Reset function for the button
def reset(event):
    p_slider.reset()

# Register the reset function with the button
reset_button.on_clicked(reset)

plt.show()
