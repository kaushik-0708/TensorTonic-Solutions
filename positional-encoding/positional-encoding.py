import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Initialize the output matrix with zeros
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Create the position vector of shape (seq_len, 1) (Hint 1)
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # Calculate the frequency divisor term based on even indices (2i)
    # np.arange(0, d_model, 2) gives [0, 2, 4, ...] up to d_model
    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)
    
    # Compute the arguments for the sin and cos functions using broadcasting
    # Shape becomes (seq_len, ceil(d_model / 2))
    args = pos / div_term
    
    # Fill even columns (0, 2, 4...) with sine values (Hint 2)
    pe[:, 0::2] = np.sin(args)
    
    # Fill odd columns (1, 3, 5...) with cosine values (Hint 2)
    # We slice `args` to `d_model // 2` to avoid dimension mismatch 
    # if d_model is an odd number (where sine has one more column than cosine).
    pe[:, 1::2] = np.cos(args[:, :d_model // 2])
    
    return pe