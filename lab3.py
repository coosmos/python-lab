import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        # Initialize parameters
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        print("Initial weights:", self.weights)
        print("Initial bias:", self.bias)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Calculate cost (MSE)
            cost = np.mean((y_predicted - y) ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (2/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
                print(f"Weights = {self.weights}, Bias = {self.bias:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def load_data(x_path, y_path):
    """Load X and y from separate CSV files"""
    # Read CSV files
    X = pd.read_csv(r'C:\Users\KIIT\Desktop\python lab\linearX.csv')
    y = pd.read_csv(r'C:\Users\KIIT\Desktop\python lab\linearY.csv')
    
    # Convert to numpy arrays
    X = X.values
    y = y.values.ravel()  # Flatten y to 1D array
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Check if the number of samples match
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y don't match!")
        
    return X, y

def plot_training_progress(model):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.cost_history)), model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

def plot_predictions(X, y_true, y_pred):
    plt.figure(figsize=(10, 6))
    for feature in range(X.shape[1]):
        plt.subplot(1, X.shape[1], feature + 1)
        plt.scatter(X[:, feature], y_true, color='blue', label='Actual', alpha=0.5)
        plt.scatter(X[:, feature], y_pred, color='red', label='Predicted', alpha=0.5)
        plt.xlabel(f'Feature {feature+1}')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # File paths
    x_path = 'x_data.csv'  # Your X features CSV file
    y_path = 'y_data.csv'  # Your y target CSV file
    
    # Load data
    X, y = load_data(x_path, y_path)
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Plot results
    plot_training_progress(model)
    plot_predictions(X, y, y_pred)
    
    # Calculate and print MSE
    mse = np.mean((y - y_pred) ** 2)
    print(f"\nMean Squared Error: {mse:.4f}")
    
    # Print final parameters
    print("\nFinal Model Parameters:")
    for i, weight in enumerate(model.weights):
        print(f"Weight {i+1}: {weight:.4f}")
    print(f"Bias: {model.bias:.4f}")