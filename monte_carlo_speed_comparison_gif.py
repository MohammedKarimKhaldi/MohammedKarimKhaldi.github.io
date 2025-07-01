import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Example (synthetic) runtimes in seconds for 10 million simulations
# Replace with real timings if available
runtimes = {
    'Python': 6.5,
    'C++ (single)': 1.2,
    'C++ (OpenMP)': 0.3
}

labels = list(runtimes.keys())
times = list(runtimes.values())
colors = ['#0074d9', '#2ecc40', '#ff4136']

fig, ax = plt.subplots(figsize=(7,5))
ax.set_xlim(0, max(times)*1.2)
ax.set_ylim(-0.5, len(labels)-0.5)
ax.set_xlabel('Runtime (seconds)')
ax.set_title('Monte Carlo Simulation Runtime Comparison')

bars = ax.barh(labels, [0]*len(labels), color=colors)

frames = 30

def animate(i):
    frac = (i+1)/frames
    for bar, t in zip(bars, times):
        bar.set_width(t * frac)
    return bars

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save('projects/cpp-implementations/monte_carlo_speed_comparison.gif', writer=PillowWriter(fps=15))
plt.close()
print('GIF saved as projects/cpp-implementations/monte_carlo_speed_comparison.gif') 