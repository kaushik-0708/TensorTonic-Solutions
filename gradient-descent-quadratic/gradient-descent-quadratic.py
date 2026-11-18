def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x = x0
    
    # Loop for the exact number of steps (Hint 1)
    for _ in range(steps):
        # 1. Compute the derivative (gradient) of ax^2 + bx + c, which is 2ax + b (Hint 2)
        gradient = 2 * a * x + b
        
        # 2. Apply the gradient descent update
        x = x - lr * gradient
        
    return float(x)