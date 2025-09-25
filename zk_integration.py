"""
Integration script to add ZK verification to existing Llama3 models
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from zk_verification import ZKVerifier
from zk_llama import ZKVerifiedLlamaForCausalLM
from llama import LlamaConfig
import json

def convert_to_zk_verified_model(model_path: str, output_path: str = None):
    """Convert a standard Llama model to ZK-verified version"""
    
    print(f"Loading model from {model_path}...")
    
    # Load the original model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    original_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Get model configuration
    config = original_model.config
    
    # Create ZK verifier
    verifier = ZKVerifier()
    
    # Create ZK-verified model
    zk_model = ZKVerifiedLlamaForCausalLM(config, verifier)
    
    # Copy weights from original model
    print("Copying weights from original model...")
    zk_model.load_state_dict(original_model.state_dict(), strict=False)
    
    # Save the ZK-verified model
    if output_path:
        print(f"Saving ZK-verified model to {output_path}...")
        torch.save({
            'model_state_dict': zk_model.state_dict(),
            'config': config,
            'verifier_config': {
                'field_size': verifier.cs.field_size,
                'generator': verifier.cs.generator
            }
        }, output_path)
    
    return zk_model, tokenizer, verifier

def run_zk_inference(model_path: str, prompt: str, enable_verification: bool = True):
    """Run inference with ZK verification"""
    
    # Convert model to ZK-verified version
    zk_model, tokenizer, verifier = convert_to_zk_verified_model(model_path)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
    
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    
    # Run inference with ZK verification
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        logits = zk_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_zk_verification=enable_verification
        )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            inference_time = 0
        
        # Get next token
        next_token_id = torch.argmax(logits[0, -1, :]).item()
        next_token = tokenizer.decode([next_token_id])
        
        print(f"Next token: {next_token}")
        print(f"Inference time: {inference_time:.4f} seconds")
    
    # Verify computations if enabled
    if enable_verification:
        print("\nVerifying computations...")
        verification_result = zk_model.verify_model_computation()
        
        # Get proof statistics
        all_proofs = zk_model.get_all_verification_proofs()
        print(f"Generated {len(all_proofs)} verification proofs")
        
        # Count proof types
        proof_types = {}
        for proof in all_proofs.values():
            proof_type = proof.proof_type
            proof_types[proof_type] = proof_types.get(proof_type, 0) + 1
        
        print("Proof type distribution:")
        for proof_type, count in proof_types.items():
            print(f"  {proof_type}: {count}")
        
        return verification_result
    
    return True

def create_zk_commitment(model_path: str, output_path: str):
    """Create ZK commitments for model weights"""
    
    print(f"Creating ZK commitments for model at {model_path}...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    verifier = ZKVerifier()
    
    # Create commitments for all model weights
    commitments = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Create commitment for this parameter
            commitment = verifier.cs.commit(param.data, hash(name) % verifier.cs.field_size)
            commitments[name] = {
                'commitment': commitment.hex(),
                'shape': list(param.shape),
                'dtype': str(param.dtype)
            }
            print(f"Created commitment for {name}: {param.shape}")
    
    # Save commitments
    with open(output_path, 'w') as f:
        json.dump(commitments, f, indent=2)
    
    print(f"Saved {len(commitments)} commitments to {output_path}")
    return commitments

def verify_model_consistency(model_path: str, commitments_path: str):
    """Verify model consistency using ZK commitments"""
    
    print(f"Verifying model consistency...")
    
    # Load model and commitments
    model = AutoModelForCausalLM.from_pretrained(model_path)
    verifier = ZKVerifier()
    
    with open(commitments_path, 'r') as f:
        commitments = json.load(f)
    
    # Verify each parameter
    verification_results = {}
    
    for name, param in model.named_parameters():
        if name in commitments and param.requires_grad:
            # Recreate commitment
            commitment = verifier.cs.commit(param.data, hash(name) % verifier.cs.field_size)
            stored_commitment = bytes.fromhex(commitments[name]['commitment'])
            
            # Verify commitment matches
            verification_results[name] = commitment == stored_commitment
            print(f"Parameter {name}: {'✓' if verification_results[name] else '✗'}")
    
    all_verified = all(verification_results.values())
    print(f"\nOverall verification: {'✓ PASSED' if all_verified else '✗ FAILED'}")
    
    return all_verified

def main():
    parser = argparse.ArgumentParser(description="ZK Verification for Llama3 Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Llama model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--output_path", type=str, help="Output path for ZK-verified model")
    parser.add_argument("--commitments_path", type=str, help="Path to save/load commitments")
    parser.add_argument("--action", type=str, choices=["inference", "convert", "commit", "verify"], 
                       default="inference", help="Action to perform")
    parser.add_argument("--disable_verification", action="store_true", help="Disable ZK verification")
    
    args = parser.parse_args()
    
    if args.action == "inference":
        run_zk_inference(
            args.model_path, 
            args.prompt, 
            enable_verification=not args.disable_verification
        )
    
    elif args.action == "convert":
        convert_to_zk_verified_model(args.model_path, args.output_path)
    
    elif args.action == "commit":
        if not args.commitments_path:
            args.commitments_path = "model_commitments.json"
        create_zk_commitment(args.model_path, args.commitments_path)
    
    elif args.action == "verify":
        if not args.commitments_path:
            args.commitments_path = "model_commitments.json"
        verify_model_consistency(args.model_path, args.commitments_path)

if __name__ == "__main__":
    main()
