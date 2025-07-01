import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

# Load spot rate data
df = pd.read_csv("spot_rates.csv")
years = df["Year"].tolist()
rates = df["SpotRate"].tolist()

# Directory for temporary frames
os.makedirs("frames", exist_ok=True)

# Create individual frames
filenames = []
for i in range(1, len(years)+1):
    plt.figure(figsize=(6, 4))
    plt.plot(years[:i], rates[:i], marker='o', linestyle='-', linewidth=2)
    plt.title("Bootstrapped Yield Curve (Progressive)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Spot Rate (%)")
    plt.ylim(0, max(rates) + 1)
    plt.grid(True)
    filename = f"frames/frame_{i:02d}.png"
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

# Create animated GIF
with imageio.get_writer("yield_curve_bootstrap.gif", mode="I", duration=0.8) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup (optional)
for filename in filenames:
    os.remove(filename)
os.rmdir("frames")

print("GIF created: yield_curve_bootstrap.gif")
