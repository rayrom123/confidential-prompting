# Zero-Knowledge Verification for Llama3 Models

This implementation adds Zero-Knowledge Proof (ZKP) verification to Llama3 models using **Sumcheck** and **Hydra** protocols. This ensures computational integrity without revealing model weights or sensitive intermediate computations.

## Features

- **Sumcheck Protocol**: Verifies sum computations in attention mechanisms
- **Hydra Protocol**: Multi-layer verification across the entire model
- **Commitment Scheme**: Cryptographic commitments for model weights
- **Attention Verification**: ZK proofs for attention computations
- **Layer-wise Verification**: Individual verification for each model layer
- **Performance Monitoring**: Benchmarking and timing for ZK operations

## Architecture

### Core Components

1. **ZKVerifier**: Main verification coordinator
2. **SumcheckProtocol**: Verifies sum-based computations
3. **HydraProtocol**: Multi-layer verification system
4. **CommitmentScheme**: Cryptographic commitments
5. **ZKVerifiedLlamaModel**: Llama model with integrated ZK verification

### Verification Flow

```
Input → Embedding → Layer1 → Layer2 → ... → LayerN → Output
  ↓         ↓         ↓        ↓              ↓        ↓
Commit → Verify → Verify → Verify → ... → Verify → Verify
```

## Installation

```bash
# Install additional dependencies for ZK verification
pip install cryptography pycryptodome
```

## Usage

### Basic Usage

```python
from zk_verification import ZKVerifier
from zk_llama import ZKVerifiedLlamaForCausalLM
from transformers import LlamaConfig

# Create ZK verifier
verifier = ZKVerifier()

# Create ZK-verified model
config = LlamaConfig(...)
model = ZKVerifiedLlamaForCausalLM(config, verifier)

# Run inference with verification
logits = model(input_ids, position_ids, enable_zk_verification=True)

# Verify all computations
is_verified = model.verify_model_computation()
```

### Command Line Interface

```bash
# Run inference with ZK verification
python zk_integration.py --action inference --model_path "meta-llama/Llama-3.2-3B-Instruct" --prompt "Hello, how are you?"

# Convert existing model to ZK-verified version
python zk_integration.py --action convert --model_path "meta-llama/Llama-3.2-3B-Instruct" --output_path "zk_verified_model.pt"

# Create ZK commitments for model weights
python zk_integration.py --action commit --model_path "meta-llama/Llama-3.2-3B-Instruct" --commitments_path "commitments.json"

# Verify model consistency using commitments
python zk_integration.py --action verify --model_path "meta-llama/Llama-3.2-3B-Instruct" --commitments_path "commitments.json"
```

### Testing

```bash
# Run comprehensive tests
python zk_example.py
```

## Protocol Details

### Sumcheck Protocol

The Sumcheck protocol verifies that a sum of values equals a target sum without revealing the individual values:

```python
# Example: Verify attention score computation
values = [q_proj_output, k_proj_output, v_proj_output]
target_sum = attention_weights
proof = verifier.sumcheck.prove_sum(values, target_sum)
```

### Hydra Protocol

The Hydra protocol enables multi-layer verification by proving consistency across layers:

```python
# Example: Verify multiple layers
layer_data = [
    (layer_id, input_tensor, output_tensor, weights)
    for layer_id, (input_tensor, output_tensor, weights) in enumerate(layers)
]
proof = verifier.hydra.prove_multi_layer(layer_data)
```

### Commitment Scheme

Cryptographic commitments ensure model weights haven't been tampered with:

```python
# Create commitment
commitment = verifier.cs.commit(weight_tensor, randomness)

# Verify commitment
is_valid = verify_commitment(commitment, weight_tensor, randomness)
```

## Verification Types

### 1. Linear Layer Verification

```python
proof = verifier.verify_linear_layer(input_tensor, weight, bias, output)
```

### 2. Attention Verification

```python
proof = verifier.verify_attention_computation(q, k, v, attention_weights, output)
```

### 3. MLP Verification

```python
proof = verifier.verify_mlp_layer(input_tensor, gate_proj, up_proj, down_proj, output)
```

### 4. RMS Norm Verification

```python
proof = verifier.verify_rms_norm(input_tensor, weight, output, eps)
```

## Performance Considerations

### Computational Overhead

- **Sumcheck**: O(log n) rounds for n values
- **Hydra**: O(l) for l layers
- **Commitments**: O(1) per parameter

### Memory Usage

- Proof storage: ~1KB per verification
- Commitment storage: ~32 bytes per parameter
- Intermediate values: 2x model size during verification

### Benchmarking

```python
# Run performance benchmark
python zk_example.py  # Includes benchmark_zk_verification()
```

## Security Properties

### Zero-Knowledge

- Model weights remain hidden during verification
- Intermediate computations are not revealed
- Only proof of correct computation is shared

### Soundness

- False positives have negligible probability
- Cryptographic commitments prevent tampering
- Cross-layer consistency ensures integrity

### Completeness

- Correct computations always pass verification
- No false negatives for valid computations

## Integration with Confidential Prompting

The ZK verification system integrates seamlessly with the existing confidential prompting framework:

```python
# Use with confidential attention
logits = model(
    input_ids=input_ids,
    position_ids=position_ids,
    confidential=True,
    enable_zk_verification=True
)
```

## Advanced Usage

### Custom Verification

```python
# Create custom verifier with specific field size
verifier = ZKVerifier(field_size=2**512)

# Disable verification for specific layers
model.layers[0].enable_zk_verification = False
```

### Proof Aggregation

```python
# Get all proofs
all_proofs = model.get_all_verification_proofs()

# Aggregate proofs
aggregated_proof = aggregate_proofs(all_proofs)
```

### Batch Verification

```python
# Verify multiple inputs simultaneously
batch_logits = model(batch_input_ids, batch_position_ids, enable_zk_verification=True)
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or disable verification for some layers
2. **Verification Failure**: Check model weights haven't been modified
3. **Performance Issues**: Use smaller field sizes or disable some verifications

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
python zk_integration.py --action inference --model_path "path/to/model" --prompt "test"
```

## Future Enhancements

- [ ] Parallel verification across multiple GPUs
- [ ] Optimized commitment schemes for large models
- [ ] Integration with existing ZK proof systems (libsnark, circom)
- [ ] Support for quantized models
- [ ] Interactive verification protocols

## References

- [Sumcheck Protocol](https://eprint.iacr.org/2016/263.pdf)
- [Hydra Protocol](https://eprint.iacr.org/2020/299.pdf)
- [Zero-Knowledge Proofs](https://en.wikipedia.org/wiki/Zero-knowledge_proof)
- [Confidential Computing](https://cloud.google.com/confidential-computing)

## License

MIT License - see LICENSE file for details.
