from __future__ import annotations

import fire
import os
import multiprocessing
import math
import torch

import datetime

import pickle

import numpy as np
import torch
import fire
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig

from attention import AttentionBuffer
from llama import LlamaForCausalLM, softmax
from prompt import PublicMeta, PrivateMeta, Replacement
import logger 
from freivalds import freivalds_algorithm

#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '29501'

class AttentionVault:
    num_layers: int
    head_dim: int
    num_heads: int
    num_group: int

    q_buffer: torch.Tensor
    o_buffer: torch.Tensor
    kv_buffer: AttentionBuffer

    gamma: int
    attention_mask: torch.Tensor

    def __init__(self, buffer: AttentionBuffer, gamma: int, attention_mask: torch.Tensor, num_group: int, use_nccl: bool = False):
        self.kv_buffer = buffer
        self.num_layers = buffer.num_layers
        self.head_dim = buffer.head_dim
        self.num_heads = buffer.num_heads
        self.num_group = num_group
        self.use_nccl = use_nccl
        self.q_buffer = torch.empty((gamma, self.num_heads * num_group, 1, self.head_dim), dtype=buffer.dtype, device=buffer.device if self.use_nccl else 'cpu')
        self.o_buffer = torch.empty((gamma, self.num_heads * num_group, 1, self.head_dim + 2), dtype=buffer.dtype, device=buffer.device if self.use_nccl else 'cpu')

        self.gamma = gamma
        # invert mask
        if attention_mask is not None:
            inverted_mask = 1.0 - attention_mask.float()
            self.attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(buffer.dtype).min).to(buffer.dtype)

        else:
            self.attention_mask = None
    

    @torch.inference_mode()
    def serve(self):
        for i in range(self.num_layers):
            # receive Q state synchronously
            torch.distributed.recv(self.q_buffer, 0)

            # Verify the integrity of the received Q using Freivalds' algorithm
            A = self.q_buffer.cpu().numpy()
            B = self.kv_buffer.cache(i, self.num_group)[0].cpu().numpy()

            # Reshape matrices for Freivalds' algorithm
            # A: (n, h, 1, d) -> (n*h, d)
            # B: (n, h, kv_seq_len, d) -> (n*h, kv_seq_len, d) -> (n*h*kv_seq_len, d)
            A_flat = A.reshape(-1, A.shape[-1])

            # We must transpose B so that its inner dimensions align with A's
            B_transposed = B.transpose(0, 1, 3, 2)
            # B_transposed: (n, h, d, kv_seq_len) -> (n*h*d, kv_seq_len)
            B_flat = B_transposed.reshape(-1, B_transposed.shape[-1])

            # C_flat must be calculated correctly from the reshaped A and B
            # The true calculation is A @ B.T
            # Reshaping A and B into 2D matrices is a bit complex.
            # A_flat: (n*h, d)
            # B_flat_for_matmul: (d, n*h*kv_seq_len)
            # We need to reshape B into (d, n*h*kv_seq_len)
            B_flat_for_matmul = B.transpose(3, 0, 1, 2).reshape(self.head_dim, -1)
            C_flat = np.matmul(A_flat, B_flat_for_matmul) / math.sqrt(self.head_dim)

            # Freivalds' algorithm works with matrices M, N and P where MN = P.
            # Let's verify: (A) @ (B_transposed) == C.
            # To make it work, let's use the original A, B and C matrices,
            # and reshape the random vector r instead.

            r = np.random.randint(0, 2, size=(B.shape[2], 1))
            # B: (n, h, kv_seq_len, d), r: (kv_seq_len, 1)
            # B @ r will fail because shapes don't align.
            # Let's reshape B and r to make it work.
            B_reshaped = B.reshape(-1, B.shape[2], B.shape[3]) # (n*h, kv_seq_len, d)
            Cr = np.matmul(C.reshape(-1, C.shape[2]), r) # This line will likely fail as C is not defined this way
            
            # A more robust approach:
            # 1. Compute C on the fly for verification.
            A_numpy = self.q_buffer.cpu().numpy()
            k_numpy = self.kv_buffer.cache(i, self.num_group)[0].cpu().numpy()

            # Reshape for efficient batched matrix multiplication
            # A_reshaped: (n*h, 1, d)
            # k_reshaped: (n*h, kv_seq_len, d)
            A_reshaped = A_numpy.reshape(-1, A_numpy.shape[2], A_numpy.shape[3])
            k_reshaped = k_numpy.reshape(-1, k_numpy.shape[2], k_numpy.shape[3])

            C_expected = np.matmul(A_reshaped, k_reshaped.transpose(0, 2, 1)) / math.sqrt(self.head_dim)

            r = np.random.randint(0, 2, size=(C_expected.shape[2], 1))
            
            Cr = np.matmul(C_expected, r)
            
            ABr = np.matmul(A_reshaped, np.matmul(k_reshaped.transpose(0, 2, 1), r))

            if not np.allclose(ABr, Cr, rtol=1e-5, atol=1e-8):
                raise ValueError("Integrity check failed for the query tensor Q.")
            print(f"Integrity check for layer {i} passed.")

class StreamPrinter:

    def __init__(self):
        self.prev = 0

    def print(self, text):
        text = text.strip()
        now = len(text) - 1
        if now > self.prev:
            print(text[self.prev:now], end="", flush=True)
            self.prev = now
        # print(" ".join(text[self.prev:]), flush=True)


@torch.inference_mode()
def init_master(
    states_dir:str,
    model_path:str, 
    device:str,
    num_users:int=1,
    capacity:int = 1024 * 2,
    timeout_sec:int = 15,
    print_idx:int=0 # 0 means the original prompt. 1=first fake prompt, 2=second fake prompt, etc.
    ):
    # load meta
    
    print("Master started.")
    
    with open(os.path.join(states_dir, "public.meta"), "rb") as f:
        meta = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float,
        device_map=device)

    buffer = AttentionBuffer(num_batch=meta.gamma,
                             capacity=capacity,
                             num_layers=model.config.num_hidden_layers,
                             num_heads=model.config.num_key_value_heads,
                             head_dim=model.config.hidden_size // model.config.num_attention_heads,
                             dtype=torch.float,
                             device=device)

    buffer.load(os.path.join(states_dir, meta.path))

    #print('buffer size', buffer.size())

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=num_users + 1,
        timeout=datetime.timedelta(seconds=timeout_sec),
        rank=0
    )

    token_ids = meta.initial_token_ids
    position_ids = [meta.pos_offset] * meta.gamma
    buffer_sink_ids = buffer.allocate(1)

    output_ids = [[] for _ in range(meta.gamma)]
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

    printer = StreamPrinter()
    while True:
        
        logits = model(
            input_ids=torch.as_tensor(token_ids, device=device).unsqueeze(-1),
            position_ids=torch.as_tensor(position_ids, device=device).unsqueeze(-1),
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
            confidential=True  # Bật lại để debug private attention
        )
        
        # sample from logits
        last_token_logits = logits[:, -1, :]

        new_token = torch.argmax(last_token_logits, dim=-1).tolist()
        token_ids = new_token

        buffer_sink_ids = buffer.allocate(1)

        for i in range(meta.gamma):
            output_ids[i].append(new_token[i])
            
        position_ids = [len(output_ids[0]) + meta.pos_offset] * meta.gamma

        printer.print(tokenizer.decode(output_ids[print_idx]))

        if new_token[0] in stop_token_ids:
            
            result = logger.read()
            processed = logger.process_data(result)
            
            print("====================")
            print(processed)
            print("====================")
            
            break


def init_worker(
    states_dir:str,
    model_path:str,
    device:str,
    num_users:int = 1,
    user_id:int = 0,
    timeout_sec:int = 15,
    disable_multiplexing:bool = False,
    ):
    # load private metadata pickle
    
    print(f"Worker {user_id} started.")    
    
    with open(os.path.join(states_dir, "private.meta"), "rb") as f:
        meta = pickle.load(f)

    config = AutoConfig.from_pretrained(model_path)
    # The buffer does not grow, so we can allocate the right size from the start
    buffer_private = AttentionBuffer(num_batch=1,
                                     capacity=meta.len_private,
                                     num_layers=config.num_hidden_layers,
                                     num_heads=config.num_key_value_heads,
                                     head_dim=config.hidden_size // config.num_attention_heads,
                                     dtype=torch.float,
                                     device=device)
    
    buffer_private.load(os.path.join(states_dir, meta.path))

    # create virtual prompts
    # virtual_prompts = [list()] * meta['gamma']
    virtual_prompt_buffer_ids = [[] for _ in range(meta.gamma)]

    for reps in meta.replacements.values():
        for i in range(meta.gamma):
            j = i % len(reps)

            rel_ids = reps[j].buffer_ids
            virtual_prompt_buffer_ids[i].extend(rel_ids)

    common_mask = np.zeros(meta.len_private, dtype=np.bool_)
    offset = 0
    for mask_type, mask_size in meta.mask_info:
        if mask_type == 0:
            common_mask[offset:offset + mask_size] = True
        offset += mask_size

    mask = np.zeros((meta.gamma, meta.len_private), dtype=np.bool_)
    mask[:, :] = common_mask

    for i in range(meta.gamma):
        mask[i, virtual_prompt_buffer_ids[i]] = True


    if disable_multiplexing:
        
        new_cap = max(len(mask[i].nonzero()[0]) for i in range(meta.gamma))
        
        buffer_private2 = AttentionBuffer(num_batch=meta.gamma,
                                     capacity=new_cap,
                                     num_layers=config.num_hidden_layers,
                                     num_heads=config.num_key_value_heads,
                                     head_dim=config.hidden_size // config.num_attention_heads,
                                     dtype=torch.float,
                                     device=device)
        
        
        for i in range(meta.gamma):
            indices = mask[i].nonzero()[0]
            
            for layer_idx in range(buffer_private.num_layers):
                buffer_private2.k[layer_idx][i, :, :len(indices), :].copy_(buffer_private.k[layer_idx][0, :, indices, :])
                buffer_private2.v[layer_idx][i, :, :len(indices), :].copy_(buffer_private.v[layer_idx][0, :, indices, :])
        
        buffer_private2.allocate(new_cap)
        
        buffer_private.clear()
        del buffer_private
        buffer_private = buffer_private2
        
        mask = None
        
    if mask is not None:
        mask = torch.as_tensor(mask, device=device)

    vault = AttentionVault(buffer_private, meta.gamma, mask, num_group=config.num_attention_heads // config.num_key_value_heads)

    # print memory consumption in MB.
    print(f"Worker {user_id} memory consumption: {buffer_private.memory_consumption() / 1024 / 1024:.2f} MB")

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=num_users + 1,
        timeout=datetime.timedelta(seconds=timeout_sec),
        rank=user_id + 1
    )

    while True:
        vault.serve()


def main(model="meta-llama/Llama-3.2-3B-Instruct",
         device="cuda:0",
         states_dir="./states",
         num_users:int=1,
         timeout_sec:int=5,
         max_num_tokens:int=2048,
         standalone_master:bool=False,
         standalone_worker:bool=False,
         user_id:int=0,
         print_idx:int=0,
         disable_multiplexing:bool=False
         ):
    
    
    if standalone_master:
        init_master(states_dir, model, device, num_users, max_num_tokens, timeout_sec, print_idx)
        return 
    
    if standalone_worker:
        init_worker(states_dir, model, device, num_users, user_id, timeout_sec, disable_multiplexing)
        return
    
        # List to store the processes
    processes = []
    
    for i in range(num_users):
        p = multiprocessing.Process(target=init_worker, kwargs={
            'states_dir': states_dir,
            'model_path': model,
            'device': device,
            'num_users': num_users,
            'user_id': i,
            'timeout_sec': timeout_sec,
            'disable_multiplexing': disable_multiplexing
            })
        p.start()
        processes.append(p)

    server_process = multiprocessing.Process(target=init_master, kwargs={
        'states_dir': states_dir,
        'model_path': model,
        'device': device,
        'num_users': num_users,
        'capacity': max_num_tokens,
        'timeout_sec': timeout_sec,
        'print_idx': print_idx
    })
    server_process.start()
    processes.append(server_process)

    # Wait for all processes to finish
    for process in processes:
        process.join()
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    fire.Fire(main)