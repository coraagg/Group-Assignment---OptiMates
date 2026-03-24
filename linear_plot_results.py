import pandas as pd
import matplotlib.pyplot as plt
import os

log_file = "logs/linear_bs64_lr0.001_losscrossentropy.csv"
df = pd.read_csv(log_file)

os.makedirs("logs/figures", exist_ok=True)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y‑axis: loss
ax1.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Right y‑axis: accuracy
ax2 = ax1.twinx()
ax2.plot(df["epoch"], df["train_acc"], marker="s", label="Train Accuracy")
ax2.plot(df["epoch"], df["val_acc"], marker="^", label="Validation Accuracy")
ax2.set_ylabel("Accuracy (%)")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Training Loss and Accuracy over Epochs")
plt.tight_layout()
plt.savefig("logs/figures/combined_metrics.png", dpi=300)
plt.show()

print("Saved: logs/figures/combined_metrics.png")
