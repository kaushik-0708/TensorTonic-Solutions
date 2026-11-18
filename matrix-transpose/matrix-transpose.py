import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0

    transposed = np.zeros((cols, rows))
    

    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = A[i][j]
            
    return transposed