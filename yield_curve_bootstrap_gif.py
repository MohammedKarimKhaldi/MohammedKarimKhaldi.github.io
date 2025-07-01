import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Synthetic bond data
maturities = np.array([1, 2, 3, 4, 5])
prices = np.array([98.0, 95.0, 92.0, 88.0, 85.0])
coupons = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
face = 100.0

spots = []
for i in range(len(maturities)):
    sum_coupons = 0.0
    for j in range(i):
        sum_coupons += coupons[i] / (1 + spots[j]) ** (j + 1)
    numer = coupons[i] + face
    denom = prices[i] - sum_coupons
    r = (numer / denom) ** (1.0 / (i + 1)) - 1.0
    spots.append(r)

frames = len(maturities)
fig, ax = plt.subplots(figsize=(7,5))
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0, max(spots)*1.5)
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Spot Rate')
ax.set_title('Bootstrapped Yield Curve')
line, = ax.plot([], [], 'bo-', lw=2)


def animate(i):
    x = maturities[:i+1]
    y = np.array(spots[:i+1])
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save('projects/cpp-implementations/yield_curve_bootstrap.gif', writer=PillowWriter(fps=1))
plt.close()
print('GIF saved as projects/cpp-implementations/yield_curve_bootstrap.gif') 