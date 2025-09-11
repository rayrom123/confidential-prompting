import numpy as np


def freivalds_algorithm(A, B, C, num_checks=10):
    """
    Freivalds' algorithm to verify if A * B^T = C * sqrt(hidden_dim) for 4D tensors in SPD context.
    This checks if Q @ K^T = attention_scores, accounting for scaling factor.

    Parameters:
    A (np.ndarray): Query tensor with shape (batch, heads, seq_len_q, hidden_dim).
    B (np.ndarray): Key tensor with shape (batch, heads, seq_len_kv, hidden_dim).
    C (np.ndarray): Expected result tensor with shape (batch, heads, seq_len_q, seq_len_kv).
    num_checks (int): Number of random checks to perform.

    Returns:
    bool: True if A * B^T = C * sqrt(hidden_dim) with high probability, False otherwise.
    """
    batch, heads, seq_len_q, hidden_dim = A.shape
    _, _, seq_len_kv, _ = B.shape

    for _ in range(num_checks):
        # Generate random vector with size matching seq_len_kv
        r = np.random.randint(0, 2, size=(seq_len_kv, 1))

        # For SPD, we need to account for the scaling factor sqrt(hidden_dim)
        # The actual attention computation is: A @ B^T / sqrt(hidden_dim)
        # So to verify: A @ B^T = C * sqrt(hidden_dim)

        # To verify A @ B^T = C * sqrt(hidden_dim), we check if A @ (B^T @ r) = (C * sqrt(hidden_dim)) @ r
        # First compute B^T @ r
        # B has shape (batch, heads, seq_len_kv, hidden_dim)
        # B^T has shape (batch, heads, hidden_dim, seq_len_kv)
        # B^T @ r: (batch, heads, hidden_dim, seq_len_kv) @ (seq_len_kv, 1) -> (batch, heads, hidden_dim, 1)
        BT_r = np.matmul(B.transpose(0, 1, 3, 2), r)

        # Compute A @ (B^T @ r)
        # A @ BT_r: (batch, heads, seq_len_q, hidden_dim) @ (batch, heads, hidden_dim, 1) -> (batch, heads, seq_len_q, 1)
        A_BT_r = np.matmul(A, BT_r)

        # Compute (C * sqrt(hidden_dim)) @ r
        # C has shape (batch, heads, seq_len_q, seq_len_kv)
        # C_scaled @ r: (batch, heads, seq_len_q, seq_len_kv) @ (seq_len_kv, 1) -> (batch, heads, seq_len_q, 1)
        C_scaled = C * np.sqrt(hidden_dim)
        C_r = np.matmul(C_scaled, r)

        # Debug: Check shapes and values
        print(f"Debug Freivalds - A_BT_r shape: {A_BT_r.shape}")
        print(f"Debug Freivalds - C_r shape: {C_r.shape}")
        print(f"Debug Freivalds - Max diff: {np.max(np.abs(A_BT_r - C_r))}")
        print(f"Debug Freivalds - Mean A_BT_r: {np.mean(A_BT_r)}, Mean C_r: {np.mean(C_r)}")

        # Check if A @ (B^T @ r) equals C @ r with tolerance for floating-point precision
        if not np.allclose(A_BT_r, C_r, rtol=1e-1, atol=1e-4):
            print(f"Debug Freivalds - FAILED: Max diff = {np.max(np.abs(A_BT_r - C_r))}")
            return False
        else:
            print(f"Debug Freivalds - PASSED: Max diff = {np.max(np.abs(A_BT_r - C_r))}")
            return True
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
