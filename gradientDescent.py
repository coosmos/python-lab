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

# Step 2: Plot the data points
plt.scatter(X, Y, color='blue', label='Data points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points')
plt.show()

# Step 3: Linear regression using gradient descent
# Add an extra column of ones to X for the intercept term (b0)
X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]  # Adding bias term

# Initialize the parameters (weights)
theta = np.zeros(2)  # Starting with zeros for b0 and b1

# Learning rate and maximum number of iterations
learning_rate = 0.005
iterations = 50
convergence_threshold = 1e-6  # Convergence threshold for cost change

# Step 4: Define the cost function (Mean Squared Error)
def compute_cost(X_b, Y, theta):
    m = len(Y)
    predictions = X_b.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - Y))
    return cost

# Step 5: Implement gradient descent with convergence criteria
def gradient_descent(X_b, Y, theta, learning_rate, iterations, convergence_threshold):
    m = len(Y)
    cost_history = np.zeros(iterations)
    prev_cost = float('inf')  # Initialize the previous cost to infinity
    
    for i in range(iterations):
        gradient = (1/m) * X_b.T.dot(X_b.dot(theta) - Y)
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X_b, Y, theta)
        
        # Check for convergence based on the change in the cost function
        cost_diff = prev_cost - cost_history[i]
        if cost_diff < convergence_threshold:
            print(f"Convergence reached at iteration {i+1}")
            break
        
        prev_cost = cost_history[i]
    
    return theta, cost_history

# Run gradient descent with convergence criteria
theta_optimal, cost_history = gradient_descent(X_b, Y, theta, learning_rate, iterations, convergence_threshold)

# Print the optimized parameters (theta) and the final cost
print("Optimal Parameters (b0, b1):", theta_optimal)
print("Final cost after convergence:", cost_history[-1])

# Step 6: Plot the best fit line
# Rescale the X values back to original scale for plotting the best fit line
X_rescaled = X_norm * np.std(X) + np.mean(X)  # Rescale to original X values

plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X_rescaled, X_b.dot(theta_optimal), color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

# Optionally, plot the cost history to observe convergence
plt.plot(range(len(cost_history)), cost_history, color='green')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
