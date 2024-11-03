import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot training and evaluation logs')
    parser.add_argument('--log_path', type=str, required=1, help='Path to the training log file')
    return parser.parse_args()

def plot_train(train_log_path):
    with open(train_log_path, 'r') as f:
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
    num_plots = len(iter_losses)
    grid_size = int(math.sqrt(num_plots)) + 1
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
    axs = axs.flatten()
    for i, (iter, losses) in enumerate(iter_losses.items()):
        if i < len(axs):
            x = range(len(losses))
            axs[i].plot(x, losses)
            axs[i].set_title(f'Iteration {iter}')
            axs[i].set_ylabel('Loss')
            axs[i].set_xlabel('Epoch')
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
    output_dir = os.path.dirname(train_log_path)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

def plot_eval(eval_log_path):
    def read_log_file(file_path):
        eval_cfr_iters = []
        baseline_avg_mbb = []
        mbb_per_hand = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' = ')
                if parts[0].startswith('eval_cfr_iter'):
                    eval_cfr_iters.append(int(parts[1]))
                elif parts[0] == 'session_baseline_total_avg_mbb':
                    baseline_avg_mbb.append(float(parts[1]))
                elif parts[0] == 'mbb_per_hand':
                    mbb_per_hand.append(float(parts[1]))
        
        return eval_cfr_iters, baseline_avg_mbb, mbb_per_hand

    eval_cfr_iters, baseline_avg_mbb, mbb_per_hand = read_log_file(eval_log_path)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot
    ax1.plot(eval_cfr_iters, baseline_avg_mbb, marker='o', linestyle='-')
    ax1.set_title('Baseline Average MBB vs. CFR Iteration')
    ax1.set_xlabel('CFR Iteration')
    ax1.set_ylabel('Baseline Average MBB')
    ax1.grid(True)
    ax1.set_xticks(eval_cfr_iters)

    # Second subplot
    ax2.plot(eval_cfr_iters, mbb_per_hand, marker='o', linestyle='-')
    ax2.set_title('MBB per Hand vs. CFR Iteration')
    ax2.set_xlabel('CFR Iteration')
    ax2.set_ylabel('MBB per Hand')
    ax2.grid(True)
    ax2.set_xticks(eval_cfr_iters)

    plt.tight_layout()
    plt.show()

def transform_line(line):
   if 'eval_cfr_iter' in line:
       return line
   
   # Replace the field names first
   line = line.replace('total_winnings', 'total_winnings_mbb')
   line = line.replace('bb_per_100', 'mbb_per_hand')
   line = line.replace('session_baseline_total_avg', 'session_baseline_total_avg_mbb')
   
   # If it's a value line, divide the number by 150
   # demark in mbb
   bb = 100
   if '=' in line:
       field, value = line.split('=')
       value = float(value.strip())
       value = value / bb * 1000
       line = f"{field.strip()} = {value}"
   
   return line

def process_file(input_text):
   output_lines = []
   for line in input_text.split('\n'):
       if line.strip():  # Skip empty lines
           output_lines.append(transform_line(line))
   return '\n'.join(output_lines)

# To use, paste the text as a string and call process_file()

def main():
    args = parse_args()
    log_path = args.log_path
    plot_eval(f'{log_path}/eval.log')
    #plot_train(f'{log_path}/train.log')

if __name__ == "__main__":
    main()