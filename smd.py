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

    def __init__(self, buffer: AttentionBuffer, gamma: int, attention_mask: torch.Tensor, num_group: int, use_nccl: bool = False,
                 freivalds: bool = False, freivalds_tol: float = 1e-4):
        self.kv_buffer = buffer
        self.num_layers = buffer.num_layers
        self.head_dim = buffer.head_dim
        self.num_heads = buffer.num_heads
        self.num_group = num_group
        self.use_nccl = use_nccl
        self.q_buffer = torch.empty((gamma, self.num_heads * num_group, 1, self.head_dim), dtype=buffer.dtype, device=buffer.device if self.use_nccl else 'cpu')
        self.o_buffer = torch.empty((gamma, self.num_heads * num_group, 1, self.head_dim + 2), dtype=buffer.dtype, device=buffer.device if self.use_nccl else 'cpu')
        # Freivalds buffers
        self.freivalds_enabled = freivalds
        self.freivalds_tol = freivalds_tol
        if self.freivalds_enabled:
            # random vector per head shared across gamma
            self.r_buffer = torch.empty((self.num_heads * num_group, self.head_dim), dtype=buffer.dtype, device=buffer.device if self.use_nccl else 'cpu')
            # server-provided projection y = Q @ r, shape (gamma, heads, 1)
            self.y_buffer = torch.empty((gamma, self.num_heads * num_group, 1), dtype=buffer.dtype, device=buffer.device if self.use_nccl else 'cpu')

        self.gamma = gamma
        # invert mask
        if attention_mask is not None:
            inverted_mask = 1.0 - attention_mask.float()
            self.attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(buffer.dtype).min).to(buffer.dtype)

        else:
            self.attention_mask = None
    

    @torch.inference_mode()
    def serve(self):
        #print('.', end='', flush=True)
        print(f"🔄 [Worker] Starting attention vault service")
        print(f"  - Number of layers: {self.num_layers}")
        print(f"  - Freivalds enabled: {self.freivalds_enabled}")
        print(f"  - Freivalds tolerance: {self.freivalds_tol}")

        for i in range(self.num_layers):
            print(f"  📡 [Worker] Processing layer {i+1}/{self.num_layers}")
            
            # Optional Freivalds challenge: send r, then receive Q and y
            if self.freivalds_enabled:
                print(f"    🔐 [Worker] Sending Freivalds challenge vector r for layer {i+1}")
                self.r_buffer.uniform_(-1.0, 1.0)
                torch.distributed.send(self.r_buffer.contiguous(), 0)

            # receive Q state synchronously
            print(f"    📥 [Worker] Receiving Q tensor from master for layer {i+1}")
            torch.distributed.recv(self.q_buffer, 0) # [n=gamma, h, q=1, d]

            if self.freivalds_enabled:
                # receive server-computed projection y = Q @ r
                print(f"    📥 [Worker] Receiving Freivalds projection y from master for layer {i+1}")
                torch.distributed.recv(self.y_buffer, 0)

            # compute local attention
            # (n, h, 1, d) -> (1, h, n, d)
            q_new = self.q_buffer.to(self.kv_buffer.device)
            k_pvt, v_pvt = self.kv_buffer.cache(i, self.num_group)  # shape: (n, h, kv_seq_len, d)

            # (n, h, q_len, d) @ (n, h, kv_seq_len, d) -> (n, h, q_len, kv_seq_len)
            score_pvt = torch.matmul(q_new, k_pvt.transpose(2, 3)) / math.sqrt(self.head_dim)

            # apply mask
            if self.attention_mask is not None:
                score_pvt = score_pvt + self.attention_mask.unsqueeze(1).unsqueeze(1)

            max_pvt, sum_pvt, score_pvt = softmax(score_pvt, dim=-1)
            
            attn_pvt = torch.matmul(score_pvt, v_pvt)

            o_buffer = torch.cat([max_pvt, sum_pvt, attn_pvt], dim=-1)

            # Verify Freivalds projection if enabled
            if self.freivalds_enabled:
                print(f"    🔍 [Worker] Verifying Freivalds projection for layer {i+1}")
                try:
                    # Compute local projection: (gamma, heads, 1, d) @ (heads, d) -> (gamma, heads, 1)
                    # Broadcast r over gamma and query length 1
                    r = self.r_buffer.to(q_new.device)
                    y_check = (q_new * r.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
                    # Compare with received y
                    y_srv = self.y_buffer.to(q_new.device)
                    
                    print(f"    - y_check shape: {y_check.shape}")
                    print(f"    - y_srv shape: {y_srv.shape}")
                    print(f"    - y_check sample: {y_check[0, 0, 0].item()}")
                    print(f"    - y_srv sample: {y_srv[0, 0, 0].item()}")
                    
                    if not torch.allclose(y_check, y_srv, atol=self.freivalds_tol, rtol=0):
                        print(f"    ❌ [Worker] Freivalds verification FAILED for layer {i+1}!")
                        print(f"    - Tolerance: {self.freivalds_tol}")
                        print(f"    - Max difference: {torch.max(torch.abs(y_check - y_srv)).item()}")
                        raise RuntimeError("Freivalds check failed for Q: communication or integrity error detected.")
                    else:
                        print(f"    ✅ [Worker] Freivalds verification PASSED for layer {i+1}")
                except Exception as e:
                    print(f"    ❌ [Worker] Freivalds verification ERROR for layer {i+1}: {e}")
                    raise

            print(f"    📤 [Worker] Sending attention results to master for layer {i+1}")
            if self.use_nccl:
                torch.distributed.send(o_buffer.contiguous(), 0)
            else:
                torch.distributed.send(o_buffer.contiguous().cpu(), 0)

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
    print_idx:int=0, # 0 means the original prompt. 1=first fake prompt, 2=second fake prompt, etc.
    freivalds:bool=False,
    standalone_master:bool=False
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

    # Only initialize distributed when not running standalone
    if not standalone_master:
        print("Initializing distributed process group...")
        print(f"  - Backend: gloo")
        print(f"  - World size: {num_users + 1} (1 master + {num_users} workers)")
        print(f"  - Master rank: 0")
        print(f"  - Timeout: {timeout_sec} seconds")

        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=num_users + 1,
            timeout=datetime.timedelta(seconds=timeout_sec),
            rank=0
        )
        
        print("✅ Distributed process group initialized successfully!")
        print(f"  - Current rank: {dist.get_rank()}")
        print(f"  - World size: {dist.get_world_size()}")
        print(f"  - Backend: {dist.get_backend()}")
    else:
        print("🚫 Running in standalone mode - no distributed communication")
        print("  - Distributed group: NOT initialized")
        print("  - Confidential mode: DISABLED")

    token_ids = meta.initial_token_ids
    position_ids = [meta.pos_offset] * meta.gamma
    buffer_sink_ids = buffer.allocate(1)

    output_ids = [[] for _ in range(meta.gamma)]
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

    printer = StreamPrinter()
    while True:
        
        if standalone_master:
            # Standalone mode: no distributed, no confidential, no freivalds
            logits = model(
                input_ids=torch.as_tensor(token_ids, device=device).unsqueeze(-1),
                position_ids=torch.as_tensor(position_ids, device=device).unsqueeze(-1),
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
                confidential=False
            )
        else:
            # Distributed mode: can use confidential and freivalds
            logits = model(
                input_ids=torch.as_tensor(token_ids, device=device).unsqueeze(-1),
                position_ids=torch.as_tensor(position_ids, device=device).unsqueeze(-1),
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
                confidential=True,
                freivalds=freivalds
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
    freivalds:bool = False,
    freivalds_tol:float = 1e-4,
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

    vault = AttentionVault(buffer_private, meta.gamma, mask, num_group=config.num_attention_heads // config.num_key_value_heads,
                           freivalds=freivalds, freivalds_tol=freivalds_tol)

    # print memory consumption in MB.
    print(f"Worker {user_id} memory consumption: {buffer_private.memory_consumption() / 1024 / 1024:.2f} MB")

    print(f"Worker {user_id} initializing distributed process group...")
    print(f"  - Backend: gloo")
    print(f"  - World size: {num_users + 1} (1 master + {num_users} workers)")
    print(f"  - Worker rank: {user_id + 1}")
    print(f"  - Timeout: {timeout_sec} seconds")

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=num_users + 1,
        timeout=datetime.timedelta(seconds=timeout_sec),
        rank=user_id + 1
    )
    
    print(f"✅ Worker {user_id} distributed process group initialized successfully!")
    print(f"  - Current rank: {dist.get_rank()}")
    print(f"  - World size: {dist.get_world_size()}")
    print(f"  - Backend: {dist.get_backend()}")

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
         distributed_master:bool=False,
         user_id:int=0,
         print_idx:int=0,
         disable_multiplexing:bool=False,
         freivalds:bool=False,
         freivalds_tol:float=1e-4
         ):
    
    
    if standalone_master:
        init_master(states_dir, model, device, num_users, max_num_tokens, timeout_sec, print_idx, freivalds, standalone_master)
        return 
    
    if standalone_worker:
        init_worker(states_dir, model, device, num_users, user_id, timeout_sec, disable_multiplexing, freivalds, freivalds_tol)
        return
    
    if distributed_master:
        init_master(states_dir, model, device, num_users, max_num_tokens, timeout_sec, print_idx, freivalds, False)
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
            'disable_multiplexing': disable_multiplexing,
            'freivalds': freivalds,
            'freivalds_tol': freivalds_tol
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
        'print_idx': print_idx,
        'freivalds': freivalds,
        'standalone_master': False
    })
    server_process.start()
    processes.append(server_process)

    # Wait for all processes to finish
    for process in processes:
        process.join()
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    fire.Fire(main)