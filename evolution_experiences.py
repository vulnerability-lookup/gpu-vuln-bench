import pandas as pd
import matplotlib.pyplot as plt

# Load the experiment data
df = pd.read_csv("data/evolution_model_generation.csv")
df = df[df["gpu_model"] == "2 x NVIDIA L40S"]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Load the dataset evolution
df_dataset = pd.read_csv("data/evolution_dataset_train_split.csv")
df_dataset["date"] = pd.to_datetime(df_dataset["date"])
df_dataset.set_index("date", inplace=True)

# Function to plot metrics with dataset size as secondary y-axis
def plot_evolution_with_dataset(metric, ylabel, title, output):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary y-axis: metric
    ax1.plot(df.index, df[metric], marker="o", linestyle="-", color="b", label=metric)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(ylabel, color="b")
    ax1.tick_params(axis='y', labelcolor="b")

    # Secondary y-axis: dataset size
    ax2 = ax1.twinx()
    ax2.plot(df_dataset.index, df_dataset["num_examples"], marker="x", linestyle="--", color="r", label="Dataset Size")
    ax2.set_ylabel("Number of Examples", color="r")
    ax2.tick_params(axis='y', labelcolor="r")

    # Title, grid, legend
    fig.suptitle(title)
    ax1.grid(True)
    fig.tight_layout()
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.xticks(rotation=45)
    plt.savefig(output)
    plt.close()

# Example: plot duration with dataset evolution
plot_evolution_with_dataset(
    "duration",
    "Duration (seconds)",
    "Evolution of Duration with Dataset Size",
    "duration_with_dataset_evolution.png",
)

# Similarly, for gpu_power, cpu_power, energy_consumed, emissions:
for metric, ylabel, output_file in [
    ("gpu_power", "GPU Power (W)", "gpu_power_with_dataset_evolution.png"),
    ("cpu_power", "CPU Power (W)", "cpu_power_with_dataset_evolution.png"),
    ("energy_consumed", "Energy Consumed (J)", "energy_consumed_with_dataset_evolution.png"),
    ("emissions", "Emissions (kg CO2)", "emissions_with_dataset_evolution.png")
]:
    plot_evolution_with_dataset(metric, ylabel, f"{metric} Evolution with Dataset Size", output_file)
