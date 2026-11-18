import numpy as np

def rmsprop_step(
    w: list, 
    g: list, 
    s: list, 
    lr: float = 0.001, 
    beta: float = 0.9, 
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Convert inputs to NumPy arrays for vectorized operations (Hint 1)
    w_arr = np.array(w, dtype=float)
    g_arr = np.array(g, dtype=float)
    s_arr = np.array(s, dtype=float)
    
    # Step 1: Update the running average of squared gradients (Hint 2)
    s_new = beta * s_arr + (1 - beta) * (g_arr * g_arr)
    
    # Step 2: Update the parameters using the new accumulator (Hint 2)
    w_new = w_arr - (lr / (np.sqrt(s_new) + eps)) * g_arr
    
    # Convert the resulting arrays back to standard Python lists to match the return type
    return w_new.tolist(), s_new.tolist()