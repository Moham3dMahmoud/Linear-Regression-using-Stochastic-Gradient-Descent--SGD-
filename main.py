import pandas as pd
import numpy as np

FILENAME = 'MultipleLR.csv'
LEARNING_RATE = 0.01
EPOCHS = 2000

def train_sgd_linear_regression(filename, learning_rate, epochs):
    try:
        data = pd.read_csv(filename, header=None).values
    except FileNotFoundError:
        return None, None, None, None

    X, y = data[:, :-1], data[:, -1]
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8
    X_norm = (X - mu) / sigma

    X_aug = np.c_[np.ones(X_norm.shape[0]), X_norm]
    
    weights = np.zeros(X_aug.shape[1])
    n_samples = X_aug.shape[0]

    for _ in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X_aug[indices], y[indices]

        for i in range(n_samples):
            xi, yi = X_shuffled[i], y_shuffled[i]
            error = np.dot(xi, weights) - yi
            weights -= learning_rate * error * xi

    return weights[1:], weights[0], mu, sigma

if __name__ == "__main__":
    final_weights, final_bias, mean_x, std_x = train_sgd_linear_regression(FILENAME, LEARNING_RATE, EPOCHS)
    
    if final_weights is not None:
        print("\n--- Final Results ---")
        print(f"Weights: {final_weights}")
        print(f"Bias: {final_bias:.4f}")
        
        test_input = np.array([93, 89, 96])
        test_norm = (test_input - mean_x) / std_x
        prediction = np.dot(test_norm, final_weights) + final_bias
        print(f"\nTest Prediction for [93, 89, 96]: {prediction:.2f} (Target: 192)")