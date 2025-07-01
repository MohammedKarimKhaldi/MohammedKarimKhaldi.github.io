import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm

S0 = 100
r = 0.05
sigma = 0.2
T = 1.0
K_values = np.linspace(60, 140, 80)

prices = []
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

for K in K_values:
    prices.append(black_scholes_call(S0, K, r, sigma, T))

fig, ax = plt.subplots(figsize=(7,5))
ax.set_xlim(K_values[0], K_values[-1])
ax.set_ylim(0, max(prices)*1.1)
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Call Option Price')
ax.set_title('Black-Scholes Call Price vs. Strike')
line, = ax.plot([], [], 'b-', lw=2)

frames = len(K_values)

def animate(i):
    line.set_data(K_values[:i+1], prices[:i+1])
    return line,

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save('projects/standard-models/option_price_vs_strike.gif', writer=PillowWriter(fps=20))
plt.close()
print('GIF saved as projects/standard-models/option_price_vs_strike.gif') 