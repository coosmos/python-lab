import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data
data_X = np.array([1, 2, 3, 4])  # Number of Ads
data_Y = np.array([150, 200, 250, 300])  # Sales

# Step 2: Calculate cost function and gradient descent
def compute_cost(theta_0, theta_1, X, Y):
    m = len(X)
    predictions = theta_0 + theta_1 * X
    errors = predictions - Y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

def gradient_descent(X, Y, theta_0, theta_1, alpha, iterations):
    m = len(X)
    for _ in range(iterations):
        predictions = theta_0 + theta_1 * X
        errors = predictions - Y

        # Update rules for theta_0 and theta_1
        theta_0 -= alpha * (1 / m) * np.sum(errors)
        theta_1 -= alpha * (1 / m) * np.sum(errors * X)

    return theta_0, theta_1

# Initial parameters and hyperparameters
theta_0 = 0  # Initial intercept
theta_1 = 0  # Initial slope
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Perform gradient descent
theta_0, theta_1 = gradient_descent(data_X, data_Y, theta_0, theta_1, alpha, iterations)
print(f"Theta_0 (Intercept): {theta_0}")
print(f"Theta_1 (Slope): {theta_1}")

# Step 3: Prediction using optimized thetas
def predict(x):
    return theta_0 + theta_1 * x

# Predict sales for 5 ads
predicted_sales = predict(5)
print(f"Predicted Sales for 5 ads: {predicted_sales}")

# Step 4: Plotting the results
plt.scatter(data_X, data_Y, color='blue', label='Actual Data')
plt.plot(data_X, predict(data_X), color='red', label='Regression Line')
plt.scatter(5, predicted_sales, color='green', label='Prediction (5 ads)')
plt.xlabel("Number of Ads")
plt.ylabel("Sales")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.show()
