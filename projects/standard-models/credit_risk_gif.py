import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Data
X = np.array([
    [50000, 10000],
    [60000, 15000],
    [35000, 8000],
    [80000, 20000],
    [120000, 30000],
    [40000, 12000],
    [70000, 10000],
    [30000, 5000],
    [90000, 25000],
    [100000, 40000]
])
y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])

model = LogisticRegression()
model.fit(X, y)

new_applicant = np.array([[45000, 11000]])
def_prob = model.predict_proba(new_applicant)[0, 1]

# Create GIF
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
bar = ax.barh(["Default Probability"], [0], color="#0074d9")
text = ax.text(0.5, 0.7, "", ha="center", va="center", fontsize=16, color="#222")
ax.set_xlabel("Probability")
ax.set_title("Estimated Probability of Default")
ax.set_yticks([])

frames = 30

def animate(i):
    frac = i / (frames - 1)
    val = def_prob * frac
    bar[0].set_width(val)
    text.set_text(f"{val:.1%}")
    return (bar[0], text)

ani = FuncAnimation(fig, animate, frames=frames, blit=True)
ani.save("credit_risk_result.gif", writer=PillowWriter(fps=15))
plt.close()
print(f"Estimated probability of default: {def_prob:.2%}")
print("GIF saved as credit_risk_result.gif") 