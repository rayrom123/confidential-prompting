"""
Example usage of Zero-Knowledge verification for Llama3 model
Demonstrates Sumcheck and Hydra protocols in action
"""

import torch
import numpy as np
from transformers import AutoTokenizer, LlamaConfig
from zk_verification import ZKVerifier, ZKProof
from zk_llama import ZKVerifiedLlamaForCausalLM
import time

def create_test_config():
    """Create a small test configuration for Llama"""
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    return config

def test_sumcheck_protocol():
    """Test Sumcheck protocol with simple computations"""
    print("=== Testing Sumcheck Protocol ===")
    
    verifier = ZKVerifier()
    
    # Test 1: Simple sum verification
    values = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([7.0, 8.0, 9.0])
    ]
    target_sum = torch.tensor([12.0, 15.0, 18.0])
    
    proof = verifier.sumcheck.prove_sum(values, target_sum)
    print(f"Sumcheck proof created: {proof.proof_type}")
    print(f"Verification result: {proof.final_verification}")
    
    # Test 2: Attention score verification
    batch_size, seq_len, hidden_size = 2, 4, 8
    q = torch.randn(batch_size, 4, seq_len, hidden_size)  # 4 attention heads
    k = torch.randn(batch_size, 2, seq_len, hidden_size)  # 2 key-value heads
    v = torch.randn(batch_size, 2, seq_len, hidden_size)  # 2 key-value heads
    
    # Compute attention
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_size)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    
    # Verify attention computation
    attention_proof = verifier.verify_attention_computation(q, k, v, attention_weights, attention_output)
    print(f"Attention verification: {attention_proof.final_verification}")
    
    return True

def test_hydra_protocol():
    """Test Hydra protocol for multi-layer verification"""
    print("\n=== Testing Hydra Protocol ===")
    
    verifier = ZKVerifier()
    
    # Test multi-layer verification
    layer_data = []
    for layer_id in range(3):
        input_tensor = torch.randn(2, 4, 8)
        output_tensor = torch.randn(2, 4, 8)
        weights = {
            'weight1': torch.randn(8, 8),
            'weight2': torch.randn(8, 8)
        }
        layer_data.append((layer_id, input_tensor, output_tensor, weights))
    
    # Prove multi-layer computation
    multi_layer_proof = verifier.hydra.prove_multi_layer(layer_data)
    print(f"Hydra multi-layer proof: {multi_layer_proof.proof_type}")
    print(f"Multi-layer verification: {multi_layer_proof.final_verification}")
    
    return True

def test_zk_llama_model():
    """Test ZK-verified Llama model"""
    print("\n=== Testing ZK-Verified Llama Model ===")
    
    # Create test configuration
    config = create_test_config()
    verifier = ZKVerifier()
    
    # Create ZK-verified model
    model = ZKVerifiedLlamaForCausalLM(config, verifier)
    
    # Create test inputs
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Model created with {config.num_hidden_layers} layers")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass with ZK verification
    start_time = time.time()
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            enable_zk_verification=True
        )
    end_time = time.time()
    
    print(f"Forward pass completed in {end_time - start_time:.4f} seconds")
    print(f"Output logits shape: {logits.shape}")
    
    # Get verification proofs
    all_proofs = model.get_all_verification_proofs()
    print(f"Generated {len(all_proofs)} verification proofs")
    
    # Verify all computations
    verification_result = model.verify_model_computation()
    print(f"Overall verification result: {verification_result}")
    
    return verification_result

def test_linear_layer_verification():
    """Test linear layer verification"""
    print("\n=== Testing Linear Layer Verification ===")
    
    verifier = ZKVerifier()
    
    # Test linear layer
    input_tensor = torch.randn(2, 4, 8)
    weight = torch.randn(16, 8)
    bias = torch.randn(16)
    
    # Compute expected output
    expected_output = torch.matmul(input_tensor, weight.T) + bias
    
    # Verify computation
    proof = verifier.verify_linear_layer(input_tensor, weight, bias, expected_output)
    print(f"Linear layer verification: {proof.final_verification}")
    
    return proof.final_verification

def test_mlp_verification():
    """Test MLP layer verification"""
    print("\n=== Testing MLP Verification ===")
    
    verifier = ZKVerifier()
    
    # Test MLP layer
    input_tensor = torch.randn(2, 4, 8)
    gate_proj = torch.randn(16, 8)
    up_proj = torch.randn(16, 8)
    down_proj = torch.randn(8, 16)
    
    # Compute expected output
    gate_output = torch.matmul(input_tensor, gate_proj.T)
    up_output = torch.matmul(input_tensor, up_proj.T)
    silu_output = gate_output * torch.sigmoid(gate_output)
    intermediate = silu_output * up_output
    expected_output = torch.matmul(intermediate, down_proj.T)
    
    # Verify computation
    proof = verifier.verify_mlp_layer(
        input_tensor, gate_proj, up_proj, down_proj, expected_output
    )
    print(f"MLP verification: {proof.final_verification}")
    
    return proof.final_verification

def test_rms_norm_verification():
    """Test RMS normalization verification"""
    print("\n=== Testing RMS Norm Verification ===")
    
    verifier = ZKVerifier()
    
    # Test RMS norm
    input_tensor = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    eps = 1e-6
    
    # Compute expected output
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + eps)
    expected_output = weight * normalized
    
    # Verify computation
    proof = verifier.verify_rms_norm(input_tensor, weight, expected_output, eps)
    print(f"RMS norm verification: {proof.final_verification}")
    
    return proof.final_verification

def benchmark_zk_verification():
    """Benchmark ZK verification performance"""
    print("\n=== ZK Verification Benchmark ===")
    
    verifier = ZKVerifier()
    config = create_test_config()
    
    # Test different model sizes
    test_cases = [
        (1, 8, 64),   # Small
        (2, 16, 128), # Medium
        (4, 32, 256), # Large
    ]
    
    for batch_size, seq_len, hidden_size in test_cases:
        print(f"\nTesting batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight = torch.randn(hidden_size * 4, hidden_size)
        
        # Benchmark linear layer verification
        start_time = time.time()
        for _ in range(10):  # Run 10 times for average
            output = torch.matmul(input_tensor, weight.T)
            proof = verifier.verify_linear_layer(input_tensor, weight, None, output)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Average verification time: {avg_time:.4f} seconds")
        print(f"Verification result: {proof.final_verification}")

def main():
    """Run all tests"""
    print("Zero-Knowledge Verification for Llama3 Model")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_sumcheck_protocol()
        test_hydra_protocol()
        test_linear_layer_verification()
        test_mlp_verification()
        test_rms_norm_verification()
        
        # Run model test
        test_zk_llama_model()
        
        # Run benchmark
        benchmark_zk_verification()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
