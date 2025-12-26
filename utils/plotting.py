import matplotlib
matplotlib.use('Agg')  # Headless-safe
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_losses(loss_history, output_path="model/training_loss.png"):
    """
    Plots the training losses over time.
    """
    if not loss_history:
        print("No loss history to plot.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract all keys present in the first element to know what to plot
    keys = list(loss_history[0].keys())

    # Filter only keys that represent losses (start with 'loss')
    loss_keys = [k for k in keys if k.startswith('loss')]

    # Prepare data for plotting
    data = {key: [] for key in loss_keys}
    for entry in loss_history:
        for key in loss_keys:
            val = entry.get(key, 0.0)
            # Handle potential tensors (though they should be floats already)
            if hasattr(val, 'item'):
                val = val.item()
            data[key].append(float(val))

    plt.figure(figsize=(14, 7), dpi=100)

    # Plot 'loss' (total loss) in red as requested - bold and thick
    if 'loss' in data:
        plt.plot(data['loss'], color='red', label='Average Loss (Total)', linewidth=3, zorder=10)

    # Plot other losses in different colors
    other_keys = [k for k in loss_keys if k != 'loss']

    # Use a nice color cycle for components
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, key in enumerate(other_keys):
        color = colors[i % len(colors)]
        plt.plot(data[key], label=key, alpha=0.8, linewidth=1.5)

    plt.title('Training Loss Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Logged Step', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close() # Important for memory
    print(f"📈 Loss plot saved to {output_path}")
