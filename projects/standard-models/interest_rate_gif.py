import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Vasicek model parameters
r0 = 0.03      # initial rate
kappa = 0.15   # mean reversion speed
theta = 0.05   # long-term mean
sigma = 0.01   # volatility
T = 2.0        # years
N = 100        # time steps
M = 10         # number of paths

dt = T / N
rates = np.zeros((M, N+1))
rates[:, 0] = r0
np.random.seed(42)

# Simulate paths
for m in range(M):
    for t in range(1, N+1):
        dr = kappa * (theta - rates[m, t-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        rates[m, t] = rates[m, t-1] + dr

times = np.linspace(0, T, N+1)

# Create GIF
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, T)
ax.set_ylim(np.min(rates)-0.01, np.max(rates)+0.01)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Interest Rate')
ax.set_title('Vasicek Model: Simulated Interest Rate Paths')
lines = [ax.plot([], [], lw=2)[0] for _ in range(M)]

frames = N+1

def animate(i):
    for m, line in enumerate(lines):
        line.set_data(times[:i+1], rates[m, :i+1])
    return lines

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save('interest_rate_paths.gif', writer=PillowWriter(fps=15))
plt.close()
print(f"Simulated {M} Vasicek interest rate paths over {T} years.\nGIF saved as interest_rate_paths.gif") 