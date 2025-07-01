import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(42)
num_assets = 4
num_portfolios = 200
returns = np.random.normal(0.12, 0.05, (1000, num_assets))
cov_matrix = np.cov(returns, rowvar=False)
mean_returns = returns.mean(axis=0)

results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets))
    weights_record.append(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_vol
    results[0,i] = port_vol
    results[1,i] = port_return
    results[2,i] = sharpe

max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[0])

fig, ax = plt.subplots(figsize=(7,5))
ax.set_xlim(0, results[0].max()*1.1)
ax.set_ylim(results[1].min()*0.9, results[1].max()*1.1)
ax.set_xlabel('Volatility (Std. Deviation)')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier (Animated)')
sc = ax.scatter([], [], c=[], cmap='viridis', marker='o', s=30, alpha=0.7)
min_vol_point, = ax.plot([], [], 'ro', markersize=10, label='Min Volatility')
max_sharpe_point, = ax.plot([], [], 'go', markersize=10, label='Max Sharpe')
ax.legend()

frames = num_portfolios

def animate(i):
    sc.set_offsets(np.c_[results[0,:i+1], results[1,:i+1]])
    sc.set_array(results[2,:i+1])
    if i >= min_vol_idx:
        min_vol_point.set_data([results[0,min_vol_idx]], [results[1,min_vol_idx]])
    if i >= max_sharpe_idx:
        max_sharpe_point.set_data([results[0,max_sharpe_idx]], [results[1,max_sharpe_idx]])
    return sc, min_vol_point, max_sharpe_point

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save('portfolio_optimization.gif', writer=PillowWriter(fps=20))
plt.close()
print('GIF saved as portfolio_optimization.gif') 