import numpy as np


def freivalds_algorithm(A, B, C, num_checks=10):
    """
    Freivalds' algorithm to verify if A * B = C.

    Parameters:
    A (np.ndarray): Matrix A.
    B (np.ndarray): Matrix B.
    C (np.ndarray): Matrix C (expected result of A * B).
    num_checks (int): Number of random checks to perform.

    Returns:
    bool: True if A * B = C with high probability, False otherwise.
    """
    n = A.shape[0]
    for _ in range(num_checks):
        # Generate a random vector
        r = np.random.randint(0, 2, size=(n, 1))
        # Compute Br and Cr
        Br = np.dot(B, r)
        Cr = np.dot(C, r)
        # Compute A(Br)
        ABr = np.dot(A, Br)
        # Check if ABr equals Cr
        if not np.array_equal(ABr, Cr):
            return False
    return True

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.dot(A, B)
    print("Verification result:", freivalds_algorithm(A, B, C))
