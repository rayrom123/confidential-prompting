import numpy as np


def freivalds_algorithm(A, B, C, num_checks=10):
    """
    Freivalds' algorithm to verify if A * B^T = C for 4D tensors in SPD context.
    This checks if Q @ K^T = attention_scores.

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

        # To verify A @ B^T = C, we check if A @ (B^T @ r) = C @ r
        # First compute B^T @ r
        # B has shape (batch, heads, seq_len_kv, hidden_dim)
        # B^T has shape (batch, heads, hidden_dim, seq_len_kv)
        # B^T @ r: (batch, heads, hidden_dim, seq_len_kv) @ (seq_len_kv, 1) -> (batch, heads, hidden_dim, 1)
        BT_r = np.matmul(B.transpose(0, 1, 3, 2), r)

        # Compute A @ (B^T @ r)
        # A @ BT_r: (batch, heads, seq_len_q, hidden_dim) @ (batch, heads, hidden_dim, 1) -> (batch, heads, seq_len_q, 1)
        A_BT_r = np.matmul(A, BT_r)

        # Compute C @ r directly
        # C @ r: (batch, heads, seq_len_q, seq_len_kv) @ (seq_len_kv, 1) -> (batch, heads, seq_len_q, 1)
        C_r = np.matmul(C, r)

        # Check if A @ (B^T @ r) equals C @ r with tolerance for floating-point precision
        if not np.allclose(A_BT_r, C_r, rtol=1e-2, atol=1e-5):
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
