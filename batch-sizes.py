import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Example parameters
dataset_size = 256
per_device_batch = 32
num_gpus = 4
effective_batch = per_device_batch * num_gpus
steps_per_epoch = dataset_size // effective_batch
epochs = 5

fig, ax = plt.subplots(figsize=(12, 3))

y = 0
height = 1

# Colors
colors = ["#8dd3c7", "#fb8072", "#bebada", "#ffffb3", "#80b1d3"]

for e in range(epochs):
    x_start = 0
    for s in range(steps_per_epoch):
        rect = mpatches.Rectangle(
            (x_start, y), effective_batch, height,
            facecolor=colors[s % len(colors)],
            edgecolor="black"
        )
        ax.add_patch(rect)
        ax.text(x_start + effective_batch/2, y + height/2, f"Step {s+1}", 
                ha='center', va='center', fontsize=8)
        x_start += effective_batch
    # Label epoch
    ax.text(x_start + 0.5, y + height/2, f" Epoch {e+1}", ha='left', va='center', fontsize=10, fontweight='bold')
    y += height + 0.2

ax.set_xlim(0, dataset_size+20)
ax.set_ylim(0, y)
ax.set_xlabel("Samples processed")
ax.set_yticks([])
ax.set_title(f"Dataset={dataset_size}, per_device_batch={per_device_batch}, GPUs={num_gpus}, steps_per_epoch={steps_per_epoch}")
plt.tight_layout()
plt.show()
