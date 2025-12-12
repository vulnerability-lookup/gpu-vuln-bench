import matplotlib.pyplot as plt
import pandas as pd


def parse_training_table(md_table: str, train_batch_size: int, epoch_durations=None):
    """
    Parse a Markdown table of training results and compute:
      - steps_per_epoch
      - samples_per_epoch
      - samples_per_sec (if epoch_durations is given)

    md_table: string containing Markdown table
    train_batch_size: int, batch size used in training
    epoch_durations: list of seconds per epoch (optional)

    Returns: DataFrame with added columns
    """
    # Split lines, ignore header separator (---)
    lines = [line.strip() for line in md_table.strip().splitlines() if line.strip()]
    lines = [
        line for line in lines if not set(line) <= {"|", ":", "-"}
    ]  # remove separator row

    # Split by '|' and strip whitespace
    rows = []
    for line in lines[1:]:  # skip header
        cells = [
            cell.strip() for cell in line.split("|")[1:-1]
        ]  # remove leading/trailing empty cells
        rows.append(cells)

    # Build DataFrame
    df = pd.DataFrame(
        rows, columns=["Training Loss", "Epoch", "Step", "Validation Loss", "Accuracy"]
    )

    # Convert numeric columns
    for col in ["Training Loss", "Epoch", "Step", "Validation Loss", "Accuracy"]:
        df[col] = pd.to_numeric(df[col])

    # Compute steps per epoch
    df["steps_per_epoch"] = df["Step"].diff().fillna(df["Step"].iloc[0])

    # Compute samples per epoch
    df["samples_per_epoch"] = df["steps_per_epoch"] * train_batch_size

    # Compute samples per second if epoch durations are provided
    if epoch_durations is not None:
        if len(epoch_durations) != len(df):
            raise ValueError("epoch_durations length must match number of epochs")
        df["samples_per_sec"] = df["samples_per_epoch"] / pd.Series(epoch_durations)

    return df


md_table_a = """
| Training Loss | Epoch | Step   | Validation Loss | Accuracy |
|:-------------:|:-----:|:------:|:---------------:|:--------:|
| 0.4999        | 1.0   | 29470  | 0.6657          | 0.7290   |
| 0.5279        | 2.0   | 58940  | 0.5911          | 0.7685   |
| 0.4775        | 3.0   | 88410  | 0.5392          | 0.7961   |
| 0.3753        | 4.0   | 117880 | 0.5125          | 0.8122   |
| 0.2537        | 5.0   | 147350 | 0.5169          | 0.8232   |
"""

md_table_b = """
| Training Loss | Epoch | Step   | Validation Loss | Accuracy |
|:-------------:|:-----:|:------:|:---------------:|:--------:|
| 0.5379        | 1.0   | 29470  | 0.6573          | 0.7358   |
| 0.5714        | 2.0   | 58940  | 0.5810          | 0.7710   |
| 0.4636        | 3.0   | 88410  | 0.5412          | 0.7918   |
| 0.4738        | 4.0   | 117880 | 0.5098          | 0.8131   |
| 0.2801        | 5.0   | 147350 | 0.5175          | 0.8230   |
"""

md_table_c = """
| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 0.627         | 1.0   | 14735 | 0.6594          | 0.7298   |
| 0.5675        | 2.0   | 29470 | 0.5780          | 0.7693   |
| 0.469         | 3.0   | 44205 | 0.5363          | 0.7930   |
| 0.4373        | 4.0   | 58940 | 0.5069          | 0.8107   |
| 0.3793        | 5.0   | 73675 | 0.5071          | 0.8173   |
"""


df_a = parse_training_table(md_table_a, train_batch_size=16)
print(df_a)

df_b = parse_training_table(md_table_b, train_batch_size=16)
print(df_b)

df_c = parse_training_table(md_table_c, train_batch_size=32)
print(df_c)


# plt.figure(figsize=(10,6))
# plt.plot(df_a['Step'], (df_a['Step']*16), marker='o', linestyle='solid', label='Experiment A (Batch Size 16)')
# plt.plot(df_b['Step'], (df_b['Step']*16), marker='+', linestyle='dashed', label='Experiment B (Batch Size 16)')
# plt.plot(df_c['Step'], (df_c['Step']*32), marker='o', label='Experiment C (Batch Size 32)')
# plt.xlabel("Cumulative Training Steps")
# plt.ylabel("Cumulative Samples Processed")
# plt.title("Cumulative Samples vs Steps")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df_a["Epoch"], df_a["Accuracy"], marker="o", label="Accuracy A")
plt.plot(df_b["Epoch"], df_b["Accuracy"], marker="o", label="Accuracy B")
plt.plot(df_c["Epoch"], df_c["Accuracy"], marker="o", label="Accuracy C")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
