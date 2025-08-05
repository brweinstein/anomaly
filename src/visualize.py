import matplotlib.pyplot as plt
import os
from src.config import DATA_DIR, GRAPH_DIR

def plot_anomalies(df, title="Stock Anomalies"):
   fig, ax = plt.subplots(figsize=(12, 6))
   ax.plot(df.index, df["Close"], label="Close Price", color="gray")

   high = df[df["direction"] == "high"]
   low = df[df["direction"] == "low"]

   ax.scatter(high.index, high["Close"], color="red", label="High Close Anomaly", s=50, alpha=0.7)
   ax.scatter(low.index, low["Close"], color="blue", label="Low Close Anomaly", s=50, alpha=0.7)

   ax.set_title(title)
   ax.legend()

   os.makedirs(GRAPH_DIR, exist_ok=True)
   filename = f"{GRAPH_DIR}/{title.replace(' ', '_')}.png"
   plt.savefig(filename)
   plt.close(fig)

   print(f"Saved plot: {filename}")