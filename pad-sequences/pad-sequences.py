import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    
    # Handle the empty sequence collection requirement
    if N == 0:
        return np.zeros((0, 0), dtype=int)
        
    # Derive max_len from the longest sequence if not provided (Hint 1)
    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len
        
    # Initialize the output with np.full using the pad_value (Hint 2)
    output = np.full((N, L), pad_value, dtype=int)
    
    # Copy each truncated sequence into its row (Hint 2)
    for i, seq in enumerate(seqs):
        # We only copy up to L elements to handle truncation for sequences longer than max_len
        valid_len = min(len(seq), L)
        output[i, :valid_len] = seq[:valid_len]
        
    return output