import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Get the number of samples (N) and features (D)
    N, D = X.shape
    
    # Initialize weights w as zeros and bias b as 0.0
    w = np.zeros(D)
    b = 0.0
    
    # Gradient descent loop
    for _ in range(steps):
        # 1. Forward pass: compute predictions
        z = X @ w + b
        p = _sigmoid(z)
        
        # 2. Compute gradients (using Hint 1)
        grad_w = X.T @ (p - y) / N
        grad_b = np.mean(p - y)
        
        # 3. Update parameters with the learning rate (using Hint 2)
        w -= lr * grad_w
        b -= lr * grad_b
        
    return (w, b)