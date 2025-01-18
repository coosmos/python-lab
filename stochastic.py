import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load data from two CSV files
X = pd.read_csv('linearX.csv', header=None).values  # Assuming one column of X data
Y = pd.read_csv('linearY.csv', header=None).values  # Assuming one column of Y data

# Flatten Y to ensure it's a 1D array
Y = Y.flatten()

# Normalize X: Subtract mean and divide by standard deviation
X_norm = (X - np.mean(X)) / np.std(X)

# Add an extra column of ones to X for the intercept term (b0)
X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]  # Adding bias term

# Initialize the parameters (weights)
theta = np.zeros(2)  # Starting with zeros for b0 and b1

# Learning rate
learning_rate = 0.01  # Choose a suitable learning rate

# Maximum number of iterations
iterations = 1000

# Batch size for mini-batch gradient descent
batch_size = 32

# Step 4: Define the cost function (Mean Squared Error)
def compute_cost(X_b, Y, theta):
    m = len(Y)
    predictions = X_b.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - Y))
    return cost

# Step 5: Implement gradient descent
def gradient_descent(X_b, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        gradient = (1 / m) * X_b.T.dot(X_b.dot(theta) - Y)
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X_b, Y, theta)

    return theta, cost_history

# Step 6: Implement stochastic gradient descent
def stochastic_gradient_descent(X_b, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        random_index = np.random.randint(m)
        X_i = X_b[random_index:random_index + 1]
        Y_i = Y[random_index]
        gradient = X_i.T.dot(X_i.dot(theta) - Y_i)  # Note: No need to divide by m here
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X_b, Y, theta)

    return theta, cost_history

# Step 7: Implement mini-batch gradient descent
def mini_batch_gradient_descent(X_b, Y, theta, learning_rate, iterations, batch_size):
    m = len(Y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X_b[indices]
        Y_batch = Y[indices]
        gradient = (1 / batch_size) * X_batch.T.dot(X_batch.dot(theta) - Y_batch)
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X_b, Y, theta)

    return theta, cost_history

# Run gradient descent
theta_batch, cost_history_batch = gradient_descent(X_b, Y, theta, learning_rate, iterations)

# Run stochastic gradient descent
theta_stochastic, cost_history_stochastic = stochastic_gradient_descent(X_b, Y, theta, learning_rate, iterations)

# Run mini-batch gradient descent
theta_mini_batch, cost_history_mini_batch = mini_batch_gradient_descent(X_b, Y, theta, learning_rate, iterations, batch_size)

# Plot cost function vs. iteration for each method
plt.plot(range(iterations), cost_history_batch, label='Batch Gradient Descent')
plt.plot(range(iterations), cost_history_stochastic, label='Stochastic Gradient Descent')
plt.plot(range(iterations), cost_history_mini_batch, label='Mini-Batch Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.legend()
plt.show()

# Print final cost for each method
print(f"Batch GD Final Cost: {cost_history_batch[-1]}")
print(f"Stochastic GD Final Cost: {cost_history_stochastic[-1]}")
print(f"Mini-Batch GD Final Cost: {cost_history_mini_batch[-1]}")