import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot training and evaluation logs')
    parser.add_argument('--log_path', type=str, required=0, help='Path to the training log file')
    return parser.parse_args()
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

    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Show every 10th tick
    tick_indices = range(0, len(eval_cfr_iters), 10)
    tick_values = [eval_cfr_iters[i] for i in tick_indices]

    # First subplot
    ax1.plot(eval_cfr_iters, baseline_avg_mbb, marker='o', linestyle='-')
    ax1.set_title('Baseline Total Milli Big Blind (MBB, or 1/1000th of Big Blind) vs. CFR Iteration')
    ax1.set_xlabel('CFR Iteration')
    ax1.set_ylabel('Baseline Total MBB')
    ax1.grid(True)
    ax1.set_xticks(tick_values)
    ax1.tick_params(axis='x', rotation=45)

    # Second subplot
    ax2.plot(eval_cfr_iters, mbb_per_hand, marker='o', linestyle='-')
    ax2.set_title('MBB per Hand vs. CFR Iteration')
    ax2.set_xlabel('CFR Iteration')
    ax2.set_ylabel('MBB per Hand')
    ax2.grid(True)
    ax2.set_xticks(tick_values)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(eval_log_path), 'plot.png'))
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

import numpy as np
def logarithmic_smoothing(x_start, x_end, t, max_steps):
    """
    Logarithmically smooths a value from x_start to x_end based on current timestep.
    
    Args:
        x_start (float): Starting value (x)
        x_end (float): Target value (y)
        t (int): Current timestep
        max_steps (int): Maximum number of timesteps
        
    Returns:
        float: Smoothed value between x_start and x_end
    """
    # Prevent division by zero and log(0)
    t = max(1, min(t, max_steps))
    
    # Calculate logarithmic decay
    decay = 1 - (np.log(t) / np.log(max_steps))
    
    # Interpolate between start and end values
    value = x_end + (x_start - x_end) * decay
    
    return value

def main():
    args = parse_args()
    log_path = args.log_path
    if not log_path:
        log_path = os.path.join('./out',sorted(os.listdir('out'))[-1])
    plot_eval(f'{log_path}/eval.log')

if __name__ == "__main__":
    main()
