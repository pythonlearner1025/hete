import matplotlib.pyplot as plt
import os

file = 'out/20241020143633/train.log'

with open(file, 'r') as f:
    lines = f.readlines()

iter_losses = dict()

losses = []
for line in lines:
    if 'CFR ITER' in line:
        iter = int(line.strip()[-1])-1 
        if iter != 0:
            iter_losses[iter] = losses
            losses = []
    elif 'Epoch' in line:
        loss_pos = line.find(':')+2
        loss = float(line[loss_pos: -1].strip())
        losses.append(loss)

for k,v in iter_losses.items():
    print(k)
    print(len(v))

import math

# Calculate grid dimensions
num_plots = len(iter_losses)
grid_size = int(math.sqrt(num_plots)) + 1

# Create a figure with subplots in a grid
fig, axs = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Plot each iteration's losses
for i, (iter, losses) in enumerate(iter_losses.items()):
    if i < len(axs):
        x = range(len(losses))
        axs[i].plot(x, losses)
        axs[i].set_title(f'Iteration {iter}')
        axs[i].set_ylabel('Loss')
        axs[i].set_xlabel('Epoch')

# Remove any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save the figure
output_dir = os.path.dirname(file)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))