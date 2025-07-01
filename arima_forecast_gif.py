import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)
t = np.arange(120)
trend = 0.05 * t
seasonal = 2 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 1, 120)
y = 10 + trend + seasonal + noise
series = pd.Series(y)

model = ARIMA(series, order=(2,1,2))
fit = model.fit()
forecast_steps = 12
forecast = fit.forecast(steps=forecast_steps)

fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(0, len(series)+forecast_steps)
ax.set_ylim(min(series.min(), forecast.min())-2, max(series.max(), forecast.max())+2)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('ARIMA Forecast vs. Actual')
actual_line, = ax.plot([], [], 'b-', label='Actual')
forecast_line, = ax.plot([], [], 'r--', label='Forecast')
ax.legend()

frames = forecast_steps+1

def animate(i):
    actual_line.set_data(np.arange(len(series)), series)
    forecast_x = np.arange(len(series), len(series)+i)
    forecast_y = forecast[:i] if i > 0 else []
    forecast_line.set_data(forecast_x, forecast_y)
    return actual_line, forecast_line

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save('projects/standard-models/arima_forecast.gif', writer=PillowWriter(fps=2))
plt.close()
print('GIF saved as projects/standard-models/arima_forecast.gif') 