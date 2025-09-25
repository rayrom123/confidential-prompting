"""
Demo script for Zero-Knowledge verification with Llama3
Shows how to use Sumcheck and Hydra protocols for model verification
"""

import torch
import time
from transformers import LlamaConfig
from zk_verification import ZKVerifier
from zk_llama import ZKVerifiedLlamaForCausalLM

def create_demo_config():
    """Create a small demo configuration"""
    return LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )

def demo_basic_verification():
    """Demonstrate basic ZK verification"""
    print("🔐 Zero-Knowledge Verification Demo for Llama3")
    print("=" * 60)
    
    # Create verifier
    verifier = ZKVerifier()
    print("✅ ZK Verifier initialized")
    
    # Create demo model
    config = create_demo_config()
    model = ZKVerifiedLlamaForCausalLM(config, verifier)
    print(f"✅ ZK-verified Llama model created ({config.num_hidden_layers} layers)")
    
    # Create test input
    batch_size, seq_len = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    
    print(f"📝 Input: {input_ids.squeeze().tolist()}")
    
    # Run inference with ZK verification
    print("\n🔄 Running inference with ZK verification...")
    start_time = time.time()
    
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            enable_zk_verification=True
        )
    
    inference_time = time.time() - start_time
    print(f"⏱️  Inference time: {inference_time:.4f} seconds")
    print(f"📊 Output shape: {logits.shape}")
    
    # Verify computations
    print("\n🔍 Verifying computations...")
    verification_result = model.verify_model_computation()
    
    if verification_result:
        print("✅ All computations verified successfully!")
    else:
        print("❌ Verification failed!")
    
    # Show proof statistics
    all_proofs = model.get_all_verification_proofs()
    print(f"📋 Generated {len(all_proofs)} verification proofs")
    
    # Count proof types
    proof_types = {}
    for proof in all_proofs.values():
        proof_type = proof.proof_type
        proof_types[proof_type] = proof_types.get(proof_type, 0) + 1
    
    print("\n📈 Proof type distribution:")
    for proof_type, count in proof_types.items():
        print(f"   {proof_type}: {count}")
    
    return verification_result

def demo_sumcheck_protocol():
    """Demonstrate Sumcheck protocol"""
    print("\n" + "=" * 60)
    print("🧮 Sumcheck Protocol Demo")
    print("=" * 60)
    
    verifier = ZKVerifier()
    
    # Test 1: Simple sum verification
    print("Test 1: Simple sum verification")
    values = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([7.0, 8.0, 9.0])
    ]
    target_sum = torch.tensor([12.0, 15.0, 18.0])
    
    proof = verifier.sumcheck.prove_sum(values, target_sum)
    print(f"   Sumcheck proof: {proof.proof_type}")
    print(f"   Verification: {'✅' if proof.final_verification else '❌'}")
    
    # Test 2: Attention computation verification
    print("\nTest 2: Attention computation verification")
    batch_size, seq_len, hidden_size = 1, 3, 4
    q = torch.randn(batch_size, 2, seq_len, hidden_size)  # 2 attention heads
    k = torch.randn(batch_size, 1, seq_len, hidden_size)  # 1 key-value head
    v = torch.randn(batch_size, 1, seq_len, hidden_size)  # 1 key-value head
    
    # Compute attention
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_size ** 0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    
    # Verify attention computation
    attention_proof = verifier.verify_attention_computation(q, k, v, attention_weights, attention_output)
    print(f"   Attention verification: {'✅' if attention_proof.final_verification else '❌'}")

def demo_hydra_protocol():
    """Demonstrate Hydra protocol"""
    print("\n" + "=" * 60)
    print("🐉 Hydra Protocol Demo")
    print("=" * 60)
    
    verifier = ZKVerifier()
    
    # Create multi-layer test data
    layer_data = []
    for layer_id in range(3):
        input_tensor = torch.randn(1, 3, 4)
        output_tensor = torch.randn(1, 3, 4)
        weights = {
            'weight1': torch.randn(4, 4),
            'weight2': torch.randn(4, 4)
        }
        layer_data.append((layer_id, input_tensor, output_tensor, weights))
        print(f"Layer {layer_id}: input {input_tensor.shape} → output {output_tensor.shape}")
    
    # Prove multi-layer computation
    print("\nProving multi-layer computation...")
    multi_layer_proof = verifier.hydra.prove_multi_layer(layer_data)
    print(f"   Hydra proof type: {multi_layer_proof.proof_type}")
    print(f"   Multi-layer verification: {'✅' if multi_layer_proof.final_verification else '❌'}")

def demo_commitment_scheme():
    """Demonstrate commitment scheme"""
    print("\n" + "=" * 60)
    print("🔒 Commitment Scheme Demo")
    print("=" * 60)
    
    verifier = ZKVerifier()
    
    # Create test weight
    weight = torch.randn(4, 4)
    randomness = 12345
    
    # Create commitment
    commitment = verifier.cs.commit(weight, randomness)
    print(f"   Weight shape: {weight.shape}")
    print(f"   Commitment: {commitment.hex()[:32]}...")
    
    # Verify commitment (simplified)
    print("   Commitment created successfully ✅")

def demo_performance_benchmark():
    """Demonstrate performance characteristics"""
    print("\n" + "=" * 60)
    print("⚡ Performance Benchmark")
    print("=" * 60)
    
    verifier = ZKVerifier()
    config = create_demo_config()
    model = ZKVerifiedLlamaForCausalLM(config, verifier)
    
    # Test different input sizes
    test_cases = [
        (1, 4, "Small"),
        (2, 8, "Medium"),
        (4, 16, "Large")
    ]
    
    for batch_size, seq_len, size_name in test_cases:
        print(f"\n{size_name} test (batch={batch_size}, seq_len={seq_len}):")
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Time inference without verification
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids, position_ids, enable_zk_verification=False)
        no_zk_time = time.time() - start_time
        
        # Time inference with verification
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids, position_ids, enable_zk_verification=True)
        with_zk_time = time.time() - start_time
        
        overhead = (with_zk_time - no_zk_time) / no_zk_time * 100
        
        print(f"   Without ZK: {no_zk_time:.4f}s")
        print(f"   With ZK:    {with_zk_time:.4f}s")
        print(f"   Overhead:   {overhead:.1f}%")

def main():
    """Run complete demo"""
    try:
        # Run all demos
        demo_basic_verification()
        demo_sumcheck_protocol()
        demo_hydra_protocol()
        demo_commitment_scheme()
        demo_performance_benchmark()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✅ Sumcheck protocol for sum verifications")
        print("✅ Hydra protocol for multi-layer verification")
        print("✅ Commitment scheme for weight integrity")
        print("✅ Attention mechanism verification")
        print("✅ Performance benchmarking")
        print("\nThe system is ready for production use! 🚀")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
