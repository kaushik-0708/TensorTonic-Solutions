import numpy as np

def dropout(x, p: float, rng=None):
    """
    Implements inverted dropout for a NumPy array.
    Returns (output, dropout_pattern).
    """
    # Convert input to a NumPy array for vectorized operations
    x_arr = np.array(x, dtype=float)
    
    # Edge case: If the drop probability is 100%, return all zeros.
    if p == 1.0:
        zeros = np.zeros_like(x_arr)
        return zeros, zeros
        
    # Use the provided rng generator, or default to np.random
    if rng is not None:
        rands = rng.random(x_arr.shape)
    else:
        rands = np.random.random(x_arr.shape)
        
    # Hint 1: Retain positions below (1 - p)
    keep_mask = rands < (1.0 - p)
    
    # Hint 2: Build the pattern with 1 / (1 - p) at retained positions
    scale = 1.0 / (1.0 - p)
    dropout_pattern = keep_mask.astype(float) * scale
    
    # Apply the scaled mask to the input
    output = x_arr * dropout_pattern
    
    return output, dropout_pattern