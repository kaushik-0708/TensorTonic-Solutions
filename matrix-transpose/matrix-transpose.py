import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    # Handle the edge case of an empty matrix just in case
    if not A or not A[0]:
        return np.zeros((0, 0))
        
    rows, cols = len(A), len(A[0])
    
    # Set up the output array with flipped dimensions (cols x rows)
    res = np.zeros((cols, rows))
    
    # Loop through and swap the coordinates
    for r in range(rows):
        for c in range(cols):
            res[c, r] = A[r][c]
            
    return res