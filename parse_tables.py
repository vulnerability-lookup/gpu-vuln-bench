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
    lines = [line for line in lines if not set(line) <= {'|', ':', '-'}]  # remove separator row

    # Split by '|' and strip whitespace
    rows = []
    for line in lines[1:]:  # skip header
        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # remove leading/trailing empty cells
        rows.append(cells)

    # Build DataFrame
    df = pd.DataFrame(rows, columns=["Training Loss", "Epoch", "Step", "Validation Loss", "Accuracy"])
    
    # Convert numeric columns
    for col in ["Training Loss", "Epoch", "Step", "Validation Loss", "Accuracy"]:
        df[col] = pd.to_numeric(df[col])

    # Compute steps per epoch
    df['steps_per_epoch'] = df['Step'].diff().fillna(df['Step'].iloc[0])
    
    # Compute samples per epoch
    df['samples_per_epoch'] = df['steps_per_epoch'] * train_batch_size

    # Compute samples per second if epoch durations are provided
    if epoch_durations is not None:
        if len(epoch_durations) != len(df):
            raise ValueError("epoch_durations length must match number of epochs")
        df['samples_per_sec'] = df['samples_per_epoch'] / pd.Series(epoch_durations)

    return df


md_table_a = """
| Training Loss | Epoch | Step   | Validation Loss | Accuracy |
|:-------------:|:-----:|:------:|:---------------:|:--------:|
| 1.1521        | 1.0   | 29952  | 1.2259          | 0.3766   |
| 1.0813        | 2.0   | 59904  | 1.1410          | 0.3766   |
| 1.1229        | 3.0   | 89856  | 1.1490          | 0.3766   |
| 1.1758        | 4.0   | 119808 | 1.1433          | 0.3766   |
| 1.1702        | 5.0   | 149760 | 1.1559          | 0.3766   |
"""

md_table_b = """
| Training Loss | Epoch | Step   | Validation Loss | Accuracy |
|:-------------:|:-----:|:------:|:---------------:|:--------:|
| 0.5653        | 1.0   | 29952  | 0.6222          | 0.7460   |
| 0.4635        | 2.0   | 59904  | 0.5755          | 0.7755   |
| 0.4721        | 3.0   | 89856  | 0.5228          | 0.8027   |
| 0.4764        | 4.0   | 119808 | 0.5117          | 0.8178   |
| 0.2367        | 5.0   | 149760 | 0.5039          | 0.8253   |
"""

md_table_c = """
| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 0.6425        | 1.0   | 14976 | 0.6224          | 0.7470   |
| 0.5791        | 2.0   | 29952 | 0.5606          | 0.7748   |
| 0.4643        | 3.0   | 44928 | 0.5173          | 0.7979   |
| 0.4282        | 4.0   | 59904 | 0.4955          | 0.8175   |
| 0.2652        | 5.0   | 74880 | 0.4857          | 0.8246   |
"""


df_a = parse_training_table(md_table_a, train_batch_size=16)
print(df_a)

df_b = parse_training_table(md_table_b, train_batch_size=16)
print(df_b)

df_c = parse_training_table(md_table_c, train_batch_size=32)
print(df_c)





# plt.figure(figsize=(10,6))
# plt.plot(df_a['Step'], (df_a['Step']*16), marker='o', label='Experiment A (BS16)')
# plt.plot(df_b['Step'], (df_b['Step']*16), marker='o', label='Experiment B (BS16)')
# plt.plot(df_c['Step'], (df_c['Step']*32), marker='o', label='Experiment C (BS32)')
# plt.xlabel("Cumulative Training Steps")
# plt.ylabel("Cumulative Samples Processed")
# plt.title("Cumulative Samples vs Steps")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(10,6))
plt.plot(df_a['Epoch'], df_a['Accuracy'], marker='o', label='Accuracy A')
plt.plot(df_b['Epoch'], df_b['Accuracy'], marker='o', label='Accuracy B')
plt.plot(df_c['Epoch'], df_c['Accuracy'], marker='o', label='Accuracy C')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
