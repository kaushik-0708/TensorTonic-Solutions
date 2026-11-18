import numpy as np

def adam_step(
    param: list, 
    grad: list, 
    m: list, 
    v: list, 
    t: int, 
    lr: float = 1e-3, 
    beta1: float = 0.9, 
    beta2: float = 0.999, 
    eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Convert standard Python lists to NumPy arrays for element-wise math
    theta = np.array(param, dtype=float)
    g = np.array(grad, dtype=float)
    m_prev = np.array(m, dtype=float)
    v_prev = np.array(v, dtype=float)
    
    # Step 1: Update First Moment (momentum of the gradients)
    m_t = beta1 * m_prev + (1 - beta1) * g
    
    # Step 2: Update Second Moment (RMSProp-like magnitude of the gradients)
    v_t = beta2 * v_prev + (1 - beta2) * (g ** 2)
    
    # Step 3: Bias Correction (using the one-based timestep 't')
    m_hat = m_t / (1 - beta1 ** t)
    v_hat = v_t / (1 - beta2 ** t)
    
    # Step 4: Parameter Update (adding eps inside the denominator as requested)
    theta_t = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return theta_t, m_t, v_t