"""
Zero-Knowledge Verified Llama Model
Integrates ZK verification into the existing Llama implementation
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Dict, Any
from llama import (
    LlamaModel, LlamaForCausalLM, LlamaDecoderLayer, 
    ConfidentialLlamaAttention, LlamaAttention, LlamaMLP, LlamaRMSNorm
)
from attention import AttentionBuffer
from zk_verification import ZKVerifier, ZKProof, ZKModelWrapper
import logger

class ZKVerifiedLlamaAttention(ConfidentialLlamaAttention):
    """Llama Attention with Zero-Knowledge verification"""
    
    def __init__(self, config, verifier: ZKVerifier):
        super().__init__(config)
        self.verifier = verifier
        self.verification_proofs = {}
    
    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None,
                buffer: AttentionBuffer | None = None,
                buffer_sink_ids: list[int] | None = None,
                layer_id: int | None = 0,
                confidential: bool = False,
                num_users: int = 1,
                enable_zk_verification: bool = True
            ):
        
        logger.start_measure()
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        bsz, q_len, _ = hidden_states.size()

        if self.pvt_buffer is None:
            self.pvt_buffer = torch.empty((bsz, self.num_heads, 1, self.head_dim + 2), dtype=torch.float)
            self.pvt_buffer_gpu = torch.empty_like(self.pvt_buffer, device=buffer.device)

        # Compute Q, K, V projections
        q_new = self.q_proj(hidden_states)
        q_new = q_new.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(q_new, seq_len=torch.max(position_ids) + 1)
        q_new = self._apply_rotary_pos_emb1(q_new, cos, sin, position_ids)
        q_new_cpu = q_new.float().cpu()
        
        # ZK Verification for Q projection
        if enable_zk_verification:
            q_proof = self.verifier.verify_linear_layer(
                hidden_states, self.q_proj.weight, None, q_new
            )
            self.verification_proofs[f'q_proj_layer_{layer_id}'] = q_proof
        
        works = []
        if confidential:
            # Distributed communication for private states
            batch_per_user = bsz // num_users
            for i in range(num_users):
                torch.distributed.isend(q_new_cpu[i * batch_per_user:(i + 1) * batch_per_user], 1 + i)
                work = torch.distributed.irecv(self.pvt_buffer[i * batch_per_user:(i + 1) * batch_per_user], 1 + i)
                works.append(work)
                
            logger.log_measure('comm_overhead1')
        
        # Compute K, V projections
        k_new = self.k_proj(hidden_states)
        v_new = self.v_proj(hidden_states)
        k_new = k_new.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        k_new = self._apply_rotary_pos_emb1(k_new, cos, sin, position_ids)

        # ZK Verification for K, V projections
        if enable_zk_verification:
            k_proof = self.verifier.verify_linear_layer(
                hidden_states, self.k_proj.weight, None, k_new
            )
            v_proof = self.verifier.verify_linear_layer(
                hidden_states, self.v_proj.weight, None, v_new
            )
            self.verification_proofs[f'k_proj_layer_{layer_id}'] = k_proof
            self.verification_proofs[f'v_proj_layer_{layer_id}'] = v_proof

        # Compute public attention
        buffer.sink(layer_id, buffer_sink_ids, k_new, v_new)
        k_pub, v_pub = buffer.cache(layer_id, self.num_key_value_groups)
        logger.log_measure('qkv_projection')

        # Attention computation
        score_pub = torch.matmul(q_new, k_pub.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            score_pub = score_pub + attention_mask

        max_pub, sum_pub, score_pub = self._softmax(score_pub, dim=-1)
        attn_pub = torch.matmul(score_pub.to(q_new.dtype), v_pub)
        
        # ZK Verification for attention computation
        if enable_zk_verification:
            attention_proof = self.verifier.verify_attention_computation(
                q_new, k_pub, v_pub, score_pub, attn_pub
            )
            self.verification_proofs[f'attention_layer_{layer_id}'] = attention_proof
        
        logger.log_measure('pub_attention')
        
        if confidential:
            # Wait for private states
            for work in works:
                work.wait()
            
            self.pvt_buffer_gpu.copy_(self.pvt_buffer)
            logger.log_measure('comm_overhead2')

            aa = self.pvt_buffer_gpu if hidden_states.dtype == torch.half else self.pvt_buffer_gpu.float()
            max_pvt, sum_pvt, attn_pvt = torch.split(aa, [1, 1, self.head_dim], dim=-1)
            
            # Combine public and private states
            alpha = torch.exp(max_pvt - max_pub)
            c_pvt = sum_pvt / (sum_pvt + sum_pub / alpha)
            c_pub = sum_pub / (sum_pub + sum_pvt * alpha)
            attn = c_pvt * attn_pvt + c_pub * attn_pub
            
            logger.log_measure('pvt_attention')
        else:
            attn = attn_pub

        # Output projection
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.reshape(bsz, q_len, self.hidden_size)
        attn = self.o_proj(attn)
        
        # ZK Verification for output projection
        if enable_zk_verification:
            o_proof = self.verifier.verify_linear_layer(
                attn.reshape(bsz, q_len, self.hidden_size), 
                self.o_proj.weight, None, attn
            )
            self.verification_proofs[f'o_proj_layer_{layer_id}'] = o_proof
        
        logger.log_measure('o_projection')
        
        return attn.to(original_dtype)
    
    def _apply_rotary_pos_emb1(self, x, cos, sin, position_ids):
        """Apply rotary position embedding"""
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def _softmax(self, x: torch.Tensor, dim: int):
        """Softmax implementation"""
        m = torch.max(x, dim=dim, keepdim=True).values
        up = torch.exp((x - m).float())
        down = torch.sum(up, dim=dim, keepdim=True)
        return m, down.to(x.dtype), (up / down).to(x.dtype)

class ZKVerifiedLlamaMLP(LlamaMLP):
    """Llama MLP with Zero-Knowledge verification"""
    
    def __init__(self, config, verifier: ZKVerifier):
        super().__init__(config)
        self.verifier = verifier
        self.verification_proofs = {}
    
    def forward(self, x, enable_zk_verification: bool = True):
        # Gate projection
        gate_output = self.gate_proj(x)
        
        # Up projection  
        up_output = self.up_proj(x)
        
        # SiLU activation
        silu_output = gate_output * torch.sigmoid(gate_output)
        
        # Element-wise multiplication
        intermediate = silu_output * up_output
        
        # Down projection
        down_output = self.down_proj(intermediate)
        
        # ZK Verification for MLP computation
        if enable_zk_verification:
            mlp_proof = self.verifier.verify_mlp_layer(
                x, self.gate_proj.weight, self.up_proj.weight, 
                self.down_proj.weight, down_output
            )
            self.verification_proofs['mlp_computation'] = mlp_proof
        
        return down_output

class ZKVerifiedLlamaRMSNorm(LlamaRMSNorm):
    """Llama RMS Norm with Zero-Knowledge verification"""
    
    def __init__(self, hidden_size, eps=1e-6, verifier: ZKVerifier = None):
        super().__init__(hidden_size, eps)
        self.verifier = verifier
        self.verification_proofs = {}
    
    def forward(self, hidden_states, enable_zk_verification: bool = True):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)
        
        # ZK Verification for RMS normalization
        if enable_zk_verification and self.verifier is not None:
            rms_proof = self.verifier.verify_rms_norm(
                hidden_states, self.weight, output, self.variance_epsilon
            )
            self.verification_proofs['rms_norm'] = rms_proof
        
        return output

class ZKVerifiedLlamaDecoderLayer(LlamaDecoderLayer):
    """Llama Decoder Layer with Zero-Knowledge verification"""
    
    def __init__(self, config, verifier: ZKVerifier):
        super().__init__(config)
        self.verifier = verifier
        
        # Replace components with ZK-verified versions
        self.self_attn = ZKVerifiedLlamaAttention(config, verifier)
        self.mlp = ZKVerifiedLlamaMLP(config, verifier)
        self.input_layernorm = ZKVerifiedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, verifier=verifier)
        self.post_attention_layernorm = ZKVerifiedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, verifier=verifier)
    
    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None,
                buffer: AttentionBuffer | None = None,
                buffer_sink_ids: list[int] | None = None,
                layer_id: int | None = 0,
                confidential: bool = False,
                num_users: int = 1,
                enable_zk_verification: bool = True
    ) -> torch.Tensor:
        
        residual = hidden_states

        # Input layer norm
        hidden_states = self.input_layernorm(hidden_states, enable_zk_verification)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
            layer_id=layer_id,
            confidential=confidential,
            num_users=num_users,
            enable_zk_verification=enable_zk_verification
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, enable_zk_verification)
        hidden_states = self.mlp(hidden_states, enable_zk_verification)
        hidden_states = residual + hidden_states

        return hidden_states

class ZKVerifiedLlamaModel(LlamaModel):
    """Llama Model with Zero-Knowledge verification"""
    
    def __init__(self, config, verifier: ZKVerifier):
        super().__init__(config)
        self.verifier = verifier
        
        # Replace decoder layers with ZK-verified versions
        self.layers = nn.ModuleList([
            ZKVerifiedLlamaDecoderLayer(config, verifier) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Replace norm with ZK-verified version
        self.norm = ZKVerifiedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, verifier=verifier)
    
    def forward(self, input_ids: torch.LongTensor,
                position_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None,
                buffer: AttentionBuffer | None = None,
                buffer_sink_ids: list[int] | None = None,
                confidential: bool = False,
                num_users: int = 1,
                enable_zk_verification: bool = True,
    ) -> torch.Tensor:
        
        batch_size, seq_length = input_ids.shape
        position_ids = position_ids.view(-1, seq_length).long()
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if buffer is not None:
            seq_length_with_past = len(buffer)
            past_key_values_length = len(buffer) - seq_length

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        else:
            inverted_mask = 1.0 - attention_mask.float()
            attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                                       torch.finfo(inputs_embeds.dtype).min)

        # Process through decoder layers with ZK verification
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
                layer_id=idx,
                confidential=confidential,
                num_users=num_users,
                enable_zk_verification=enable_zk_verification
            )
            hidden_states = layer_outputs

        # Final normalization with ZK verification
        hidden_states = self.norm(hidden_states, enable_zk_verification)

        return hidden_states
    
    def get_verification_proofs(self) -> Dict[str, ZKProof]:
        """Get all verification proofs from all layers"""
        all_proofs = {}
        for i, layer in enumerate(self.layers):
            layer_proofs = {}
            layer_proofs.update(layer.self_attn.verification_proofs)
            layer_proofs.update(layer.mlp.verification_proofs)
            layer_proofs.update(layer.input_layernorm.verification_proofs)
            layer_proofs.update(layer.post_attention_layernorm.verification_proofs)
            
            for key, proof in layer_proofs.items():
                all_proofs[f'layer_{i}_{key}'] = proof
        
        # Add final norm proofs
        all_proofs.update(self.norm.verification_proofs)
        
        return all_proofs

class ZKVerifiedLlamaForCausalLM(LlamaForCausalLM):
    """Llama For Causal LM with Zero-Knowledge verification"""
    
    def __init__(self, config, verifier: ZKVerifier):
        super().__init__(config)
        self.model = ZKVerifiedLlamaModel(config, verifier)
        self.verifier = verifier
        self.verification_proofs = {}
    
    def forward(self, input_ids: torch.LongTensor,
                position_ids: torch.LongTensor,
                attention_mask: torch.Tensor | None = None,
                buffer: AttentionBuffer | None = None,
                buffer_sink_ids: list[int] | None = None,
                confidential: bool = False,
                num_users: int = 1,
                enable_zk_verification: bool = True,
    ) -> torch.Tensor:
        
        # Get hidden states from model
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
            confidential=confidential,
            num_users=num_users,
            enable_zk_verification=enable_zk_verification
        )

        # Generate logits
        logits = self.lm_head(hidden_states).float()
        
        # ZK Verification for final linear layer
        if enable_zk_verification:
            lm_head_proof = self.verifier.verify_linear_layer(
                hidden_states, self.lm_head.weight, None, logits
            )
            self.verification_proofs['lm_head'] = lm_head_proof
        
        return logits
    
    def get_all_verification_proofs(self) -> Dict[str, ZKProof]:
        """Get all verification proofs from model and layers"""
        all_proofs = {}
        all_proofs.update(self.model.get_verification_proofs())
        all_proofs.update(self.verification_proofs)
        return all_proofs
    
    def verify_model_computation(self) -> bool:
        """Verify all model computations"""
        all_proofs = self.get_all_verification_proofs()
        
        for proof_name, proof in all_proofs.items():
            if not proof.final_verification:
                print(f"Verification failed for {proof_name}")
                return False
        
        print("All model computations verified successfully!")
        return True
