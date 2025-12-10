import json
import sys
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]

"""
Usage:

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python compare_experiments.py data/2_NVIDIA_L40S.csv data/2_NVIDIA_H100_NVL.csv data/4_NVIDIA_L40S.csv

pandoc experiments_report.md -f markdown-implicit_figures --toc --number-sections -t pdf -o gpu-performance-vuln-model.pdf
"""

def load_csv(file_path):
    """Load a single-row CodeCarbon CSV."""
    df = pd.read_csv(file_path)
    if len(df) != 1:
        raise ValueError(f"CSV {file_path} should contain exactly one row")
    return df.iloc[0]

def plot_bar(labels, values, ylabel, title, output, color="skyblue"):
    """Generates a bar chart with values labeled above the bars."""
    plt.figure(figsize=(12, 9))
    bars = plt.bar(labels, values, color=color)

    # Add labels above each bar
    for bar, value in zip(bars, values):
        height = bar.get_height()
        # Ensure values are plotted even if they are close to zero
        text_height = max(height, 0.01) 
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            text_height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right') # Improved readability for long labels
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_stacked_bar(labels, cpu_energy, gpu_energy, ram_energy, ylabel, title, output):
    """Plot a stacked bar chart of energy consumption by component, with value labels inside each segment AND the total above the bar."""
    
    # Ensure all lists are of the same length for zipping
    if not (len(cpu_energy) == len(gpu_energy) == len(ram_energy)):
        raise ValueError("Energy lists must have the same length.")

    plt.figure(figsize=(12, 9))
    
    # Calculate the bottom positions for stacking
    cpu_gpu_sum = [c + g for c, g in zip(cpu_energy, gpu_energy)]
    
    # Calculate the total height for the overall label
    total_energy = [c + g + r for c, g, r in zip(cpu_energy, gpu_energy, ram_energy)]
    
    # 1. Plot CPU Energy
    bars_cpu = plt.bar(labels, cpu_energy, label='CPU Energy', color='#8dd3c7')
    
    # 2. Plot GPU Energy
    bars_gpu = plt.bar(labels, gpu_energy, bottom=cpu_energy, label='GPU Energy', color='#fb8072')
    
    # 3. Plot RAM Energy
    bars_ram = plt.bar(labels, ram_energy, bottom=cpu_gpu_sum, label='RAM Energy', color='#bebada')
    
    # --- Add Labels Inside Segments (from previous step) ---
    all_bars = [bars_cpu, bars_gpu, bars_ram]
    all_values = [cpu_energy, gpu_energy, ram_energy]
    bottom_positions = [[0] * len(labels), cpu_energy, cpu_gpu_sum] 

    for bar_set, values, bottoms in zip(all_bars, all_values, bottom_positions):
        for bar, value, bottom in zip(bar_set, values, bottoms):
            height = bar.get_height()
            
            # Only print a label if the value is non-zero and large enough
            if value > 0.01:
                # Center the text vertically within its segment
                y_position = bottom + height / 2
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_position,
                    f"{value:.2f}",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black'
                )

    # --- Add Total Value Above the Stacked Bar ---
    for i, total in enumerate(total_energy):
        if total > 0.01:
            # Place the total label right above the top of the RAM bar
            plt.text(
                i, # X-position (index of the bar)
                total, # Y-position (the total height of the stack)
                f"Total: {total:.2f}",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold', # Make the total stand out
                color='black'
            )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Energy Source", loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_scatter(x, y, labels, colors, xlabel, ylabel, title, output):
    """Generates a scatter plot with labeled points."""
    plt.figure(figsize=(12, 9))
    plt.scatter(x, y, c=colors, s=100, alpha=0.7) # Increased marker size for visibility

    # Add labels next to each point
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(5, 5))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6) # Add grid for better reading
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_radar_comparison(experiments):
    """Generate a single radar chart comparing CPU/GPU/RAM energy breakdown for all experiments."""
    categories = ["CPU (kWh)", "GPU (kWh)", "RAM (kWh)"]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for e in experiments:
        values = [e["cpu_energy"], e["gpu_energy"], e["ram_energy"]]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=e["label"])
        ax.fill(angles, values, alpha=0.25)

    plt.xticks(angles[:-1], categories)
    plt.title("CPU/GPU/RAM Energy Breakdown per Experiment (kWh)", fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # Avoid overlapping

    plt.tight_layout()
    plt.savefig("radar_energy_comparison.png")
    plt.close()



def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_experiments.py exp1.csv exp2.csv [exp3.csv ...]")
        sys.exit(1)

    # Set a modern style for better aesthetics
    plt.style.use('ggplot') 

    csv_files = sys.argv[1:]

    # ---------- LOAD EXPERIMENTS ----------
    experiments = []
    for f in csv_files:
        row = load_csv(f)
        
        label_stem = Path(f).stem
        cleaned_label = label_stem.replace("_", " ")
        gpu_model = row.get("gpu_model", "Unknown GPU")

        # Load data, using .get() with a default float value for robustness
        experiments.append({
            "label": cleaned_label,
            "gpu_model": gpu_model,
            "duration": float(row.get("duration", 0.0)),
            "gpu_power": float(row.get("gpu_power", 0.0)),
            "energy_consumed": float(row.get("energy_consumed", 0.0)),
            "emissions": float(row.get("emissions", 0.0)),
            # Energy breakdown metrics
            "cpu_energy": float(row.get("cpu_energy", 0.0)),
            "gpu_energy": float(row.get("gpu_energy", 0.0)),
            "ram_energy": float(row.get("ram_energy", 0.0)),
        })

    # Extract arrays
    labels = [e["label"] for e in experiments]
    durations = [e["duration"] for e in experiments]
    gpu_power = [e["gpu_power"] for e in experiments]
    energy = [e["energy_consumed"] for e in experiments]
    emissions = [e["emissions"] for e in experiments]
    gpu_models = [e["gpu_model"] for e in experiments]
    
    # Energy breakdown arrays
    cpu_energy = [e["cpu_energy"] for e in experiments]
    gpu_energy = [e["gpu_energy"] for e in experiments]
    ram_energy = [e["ram_energy"] for e in experiments]


    # ---------- COLORS BY GPU MODEL ----------
    unique_gpus = list(dict.fromkeys(gpu_models))  # preserve order
    palette = plt.get_cmap("Set1") # Using a more distinct palette

    color_map = {gpu: palette(i % 9) for i, gpu in enumerate(unique_gpus)}
    colors = [color_map[g] for g in gpu_models]

    # ---------- BAR CHARTS ----------
    print("Generating Bar Charts...")
    plot_bar(labels, durations,
             ylabel="Duration (seconds)",
             title="Experiment Duration Comparison",
             output="duration_comparison.png",
             color='#a6cee3') # Pastel color

    plot_bar(labels, gpu_power,
             ylabel="Avg GPU Power (W)",
             title="Average GPU Power Comparison",
             output="gpu_power_comparison.png",
             color='#1f78b4') 

    plot_bar(labels, energy,
             ylabel="Total Energy Consumed (kWh)",
             title="Energy Consumption Comparison",
             output="energy_consumption_comparison.png",
             color='#b2df8a') 
    
    # Emissions Bar Chart
    plot_bar(labels, emissions,
             ylabel="Total Emissions (kg CO2eq)",
             title="Emissions Comparison",
             output="emissions_comparison.png",
             color='#33a02c') 

    # ---------- STACKED BAR CHART (Energy Breakdown) ----------
    print("Generating Stacked Bar Chart...")
    plot_stacked_bar(labels, cpu_energy, gpu_energy, ram_energy,
                     ylabel="Energy Consumed (kWh)",
                     title="Energy Consumption Breakdown by Component",
                     output="energy_breakdown_comparison.png")

    # ---------- SCATTER CHARTS ----------
    print("Generating Scatter Charts...")
    plot_scatter(gpu_power, durations, labels, colors,
                 xlabel="Avg GPU Power (W)",
                 ylabel="Duration (seconds)",
                 title="GPU Power vs Duration",
                 output="scatter_gpu_power_vs_duration.png")

    plot_scatter(durations, energy, labels, colors,
                 xlabel="Duration (seconds)",
                 ylabel="Energy Consumed (kWh)",
                 title="Energy Consumed vs Duration",
                 output="scatter_energy_vs_duration.png")

    plot_scatter(gpu_power, energy, labels, colors,
                 xlabel="Avg GPU Power (W)",
                 ylabel="Energy Consumed (kWh)",
                 title="GPU Power vs Energy Consumed",
                 output="scatter_gpu_power_vs_energy.png")


    # ---------- RADAR CHARTS ----------
    print("Generating Radar Comparison Chart for CPU/GPU/RAM Energy Breakdown...")
    plot_radar_comparison(experiments)


    # ---------- JSON SUMMARY ----------
    print("Generating JSON Summary...")
    summary = []
    for e in experiments:
        # Avoid division by zero if duration or energy is 0
        energy_per_second = e["energy_consumed"] / e["duration"] if e["duration"] else 0.0
        seconds_per_kwh = e["duration"] / e["energy_consumed"] if e["energy_consumed"] else 0.0
        
        summary.append({
            "label": e["label"],
            "gpu_model": e["gpu_model"],
            "duration_seconds": e["duration"],
            "gpu_power_watts": e["gpu_power"],
            "energy_kwh": e["energy_consumed"],
            "emissions_kg_co2eq": e["emissions"],
            "energy_breakdown_kwh": {
                "cpu": e["cpu_energy"],
                "gpu": e["gpu_energy"],
                "ram": e["ram_energy"],
            },
            "metrics": {
                "energy_per_second_kwh": energy_per_second,
                "seconds_per_kwh": seconds_per_kwh,
            }
        })

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n--- ✅ Processing Complete ---")
    print("Charts generated:")
    print("  - duration_comparison.png")
    print("  - gpu_power_comparison.png")
    print("  - energy_consumption_comparison.png")
    print("  - emissions_comparison.png")
    print("  - energy_breakdown_comparison.png")
    print("  - scatter_gpu_power_vs_duration.png")
    print("  - scatter_energy_vs_duration.png")
    print("  - scatter_gpu_power_vs_energy.png")
    print("\nSummary written to summary.json")

if __name__ == "__main__":
    main()