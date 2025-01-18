import numpy as np
import pandas as pd

# Load data
X = pd.read_csv('linearX.csv', header=None).values
y = pd.read_csv('linearY.csv', header=None).values

# Calculate correlation coefficient
correlation = np.corrcoef(X.flatten(), y.flatten())[0,1]
print(f"Correlation coefficient between X and Y: {correlation}")

# Print the slope from our normalized model
X_normalized = (X - np.mean(X)) / np.std(X)
X_normalized = np.c_[np.ones(X.shape[0]), X_normalized]
theta = np.zeros((2, 1))
learning_rate = 0.5
num_iterations = 1000

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        predictions = X.dot(theta)
        theta[0] = theta[0] - (learning_rate/m) * np.sum(predictions - y)
        theta[1] = theta[1] - (learning_rate/m) * np.sum((predictions - y) * X[:, 1])
    return theta

final_theta = gradient_descent(X_normalized, y, theta, learning_rate, num_iterations)
print(f"\nSlope of the line (theta1): {final_theta[1][0]}")