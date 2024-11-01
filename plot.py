import matplotlib.pyplot as plt
import os

def plot_train():
    file = 'out/20241101154329/train.log'

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

def plot_eval():
    # visualize softmax
    # Define the function to read the log file and extract data
    def read_log_file(file_path):
        eval_cfr_iters = []
        bb_per_100s = []
        
        # Open and read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' = ')
                if parts[0].startswith('eval_cfr_iter'):
                    eval_cfr_iters.append(int(parts[1]))
                elif parts[0] == 'bb_per_100':
                    bb_per_100s.append(float(parts[1]))
        
        return eval_cfr_iters, bb_per_100s

    # Path to the log file
    log_file_path = 'out/20241101154329/eval.log'

    # Get the data
    eval_cfr_iters, bb_per_100s = read_log_file(log_file_path)

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(eval_cfr_iters, bb_per_100s, marker='o', linestyle='-')
    plt.title('bb_per_100 vs. eval_cfr_iter')
    plt.xlabel('eval_cfr_iter')
    plt.ylabel('bb_per_100')
    plt.grid(True)
    plt.xticks(eval_cfr_iters)  # Ensure all eval_cfr_iters are marked on the x-axis
    plt.show()


plot_eval()