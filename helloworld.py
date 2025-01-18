import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = pd.read_csv('linearX.csv', header=None).values
y = pd.read_csv('linearY.csv', header=None).values.ravel()

# Data analysis
print("X statistics:")
print(f"Mean: {np.mean(X):.4f}")
print(f"Std: {np.std(X):.4f}")
print(f"Min: {np.min(X):.4f}")
print(f"Max: {np.max(X):.4f}")

print("\ny statistics:")
print(f"Mean: {np.mean(y):.4f}")
print(f"Std: {np.std(y):.4f}")
print(f"Min: {np.min(y):.4f}")
print(f"Max: {np.max(y):.4f}")

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Data Distribution')
plt.grid(True)

# Calculate and plot regression line
slope, intercept = np.polyfit(X.ravel(), y, 1)
x_range = np.array([np.min(X), np.max(X)])
plt.plot(x_range, slope * x_range + intercept, 'r', label=f'y = {slope:.6f}x + {intercept:.6f}')
plt.legend()
plt.show()

print(f"\nRegression line equation: y = {slope:.6f}x + {intercept:.6f}")
print(f"Correlation coefficient: {np.corrcoef(X.ravel(), y)[0,1]:.6f}")