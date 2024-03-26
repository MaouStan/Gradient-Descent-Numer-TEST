from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

# Generate linear data
np.random.seed(0)
X_linear = np.random.rand(100, 1) * 10
y_linear = 2 * X_linear.squeeze() + np.random.randn(100)  # Linear relationship with noise

# Create a DataFrame
linear_df = pd.DataFrame({'X': X_linear.squeeze(), 'y': y_linear})

# Save DataFrame to CSV
linear_df.to_csv('data/linear_data.csv', index=False)

# Generate logistic data
np.random.seed(0)
X_logistic = np.random.rand(100, 2) * 10
coefficients = np.array([1, -1])  # Coefficients for logistic regression
intercept = -5  # Intercept for logistic regression
logit = np.dot(X_logistic, coefficients) + intercept
probabilities = 1 / (1 + np.exp(-logit))
y_logistic = np.random.binomial(1, probabilities)  # Generate binary labels

# Create a DataFrame
logistic_df = pd.DataFrame({'X1': X_logistic[:, 0], 'X2': X_logistic[:, 1], 'y': y_logistic})

# Save DataFrame to CSV
logistic_df.to_csv('data/logistic_data.csv', index=False)


# Generate neural network data
X_neural, y_neural = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

# Create a DataFrame
neural_df = pd.DataFrame(X_neural, columns=[f"feature_{i+1}" for i in range(X_neural.shape[1])])
neural_df['target'] = y_neural

# Save DataFrame to CSV
neural_df.to_csv('data/neural_data.csv', index=False)
