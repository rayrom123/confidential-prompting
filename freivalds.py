import numpy as np


def freivalds_algorithm(A, B, C, num_checks=10):
    """
    Freivalds' algorithm to verify if A * B^T = C for 4D tensors in SPD context.

    Parameters:
    A (np.ndarray): Query tensor with shape (batch, heads, seq_len_q, hidden_dim).
    B (np.ndarray): Key tensor with shape (batch, heads, seq_len_kv, hidden_dim).
    C (np.ndarray): Expected result tensor with shape (batch, heads, seq_len_q, seq_len_kv).
    num_checks (int): Number of random checks to perform.

    Returns:
    bool: True if A * B^T = C with high probability, False otherwise.
    """
    batch, heads, seq_len_q, hidden_dim = A.shape
    _, _, seq_len_kv, _ = B.shape

    for _ in range(num_checks):
        # Generate random vector with size matching seq_len_kv
        r = np.random.randint(0, 2, size=(seq_len_kv, 1))

        # Compute Br = B @ r (for each batch and head)
        # Shape: (batch, heads, seq_len_kv, hidden_dim) @ (seq_len_kv, 1) -> (batch, heads, hidden_dim, 1)
        Br = np.matmul(B, r)

        # Compute Cr = C @ r (for each batch and head)
        # Shape: (batch, heads, seq_len_q, seq_len_kv) @ (seq_len_kv, 1) -> (batch, heads, seq_len_q, 1)
        Cr = np.matmul(C, r)

        # Compute A @ Br
        # Shape: (batch, heads, seq_len_q, hidden_dim) @ (batch, heads, hidden_dim, 1) -> (batch, heads, seq_len_q, 1)
        ABr = np.matmul(A, Br)

        # Check if ABr equals Cr with tolerance for floating-point precision
        if not np.allclose(ABr, Cr, rtol=1e-5, atol=1e-8):
            return False
    return True

# Example usage
if __name__ == "__main__":
    # Create example tensors with 4D shapes
    batch, heads, seq_len_q, hidden_dim = 1, 2, 3, 4
    seq_len_kv = 5

    A = np.random.randn(batch, heads, seq_len_q, hidden_dim)
    B = np.random.randn(batch, heads, seq_len_kv, hidden_dim)
    # Compute C = A @ B^T
    C = np.matmul(A, B.transpose(0, 1, 3, 2))

    print("Verification result:", freivalds_algorithm(A, B, C))
