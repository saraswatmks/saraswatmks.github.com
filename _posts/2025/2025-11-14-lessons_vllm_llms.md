---
title: "5 Lessons from Deploying LLMs in Production using vLLM"
date: 2025-11-14
tags:
- llm
- vllm
excerpt: "Self learned lessons on deploying large language models in productions using vLLM"
header:
  teaser: /assets/images/lessons_learnt_llm.png
  overlay_image: /assets/images/lessons_learnt_llm.png
  overlay_filter: 0.5
---

# 5 Lessons from Deploying LLMs in Production using vLLM

Deploying an LLM has become quite easy these days. With libraries like vLLM, tensorRT-LLM etc you can deploy a huggingface or local model in few minutes. It's not a challenging task anymore.

The real challenge is still figuring out how to scale your deployed LLMs on your own hardware (H100, A100, A40 GPUs). Having deployed tiny to large LLMs, I have gone through the struggle of doing numerous deployments, optimisations, load tests to find a scalable LLM serving configuration. 

In this post, I'll share some learning and tips which I have learnt after deploying (and failing several times) LLMs using [vLLM](https://github.com/vllm-project/vllm) framework. This post is meant for people who are interested in deploying LLMs on their own hardware.


## Table of Contents

1. [Know your hardware well](#1-know-your-hardware-well)
2. [Know your business size](#2-know-your-business-size)
3. [Batch Size Follows Concurrency](#3-batch-size-follows-concurrency)
4. [Choose your Metrics](#4-choose-your-metrics)
5. [Plan Sequential LLM Calls](#5-plan-sequential-llm-calls)
6. [Summary](#6-summary)

---

### 1. Know your hardware well

Don't you ever wonder what's the best performance you can get on your hardware? What if you just knew the optimal value of LLM serving parameters? This happens with me a lot!

Let's take an example.

Imagine I am using a Gemma 27B model (with FP16 weights) on A100 80GB GPU. 

With FP16, we know each weight take 2 bytes, so the total memory needed by the model is: 27B * 2 bytes = 54GB. Ok, so far we have already used 67% of the hardware memory. 

We are left with 80GB - 54GB - (6GB overhad) = 20GB. We have this memory available for KV cache. In order to estimate KV cache size, we must know the architechture details of the gemma model. 

Lets make a few assumptions about the model parameters: 

```
n_layers = 42 (refers to no. of transformer blocks)  
n_heads = 32  
head_dim = 128  
hidden_size = n_heads * head_dim = 4096  
dtype_size = 2 (FP16 uses 2 bytes)   
```

Now, we'll find out KV cache size per token.  
```
kv_per_token = 2 * n_heads * head_dim * n_layers * dtype_size  
             = 2 * 32 * 128 * 42 * 2  
             = 688,128 bytes (688 KB approx.)  
```
Here we multipied by 2 because the KV cache consists of two matrices: Keys and Values.  

Remember KV cache matrix is stored per token. What are these tokens ? They are nothing but your input prompt + generated tokens.   

Lets assume, our input prompt contains 500 tokens. So, 1 input sequence contains 500 tokens. Lets say, for generation, you have set max_tokens=50. 

For 1 sequence, KV cache calculation becomes: 

```
kv_cache_per_sequence = 688KB x (500+50) = 378MB 
```

If you remember, we have 20GB left for KV cache, so in total by allocating all 20GB for KV cache we can get: 
```
max_sequences_possible = 20GB / 378MB = 58 sequences  
```

Now, we know given our hardware, we can process 58 requests in parallel. Lets say, processing 1 sequence takes 5 seconds, then we can calculate RPS:

```
RPS = 59 sequences/ 5 seconds = 11.8 requests/second
```

Now, you are in a better position to argue with your management, in case you need higher capacity hardware for scaling.



---

### 2. Know your business size

GPU computation is expensive ($). Knowing how much load to handle, can give you a lot of hints about what server configuration to start with. Don't try to max out the theoritical max capacity of the GPU servers. That's not optimal.

For example: In vLLM:
- `--max-num-seqs` refers to number of concurrent requests to give to the GPU. 
- `--max-num-batched-tokens` refers to the total overall number of tokens (including prompt + generated tokens) to process in one batch

For instance, if you are aiming for 10rps and you set the following configuration:
`--max-num-seqs 256 --max-num-batched-tokens 32768` 

You are heavily **underutilising** the resources here. Its like you have a 256 seat bus but you are only carrying 10 passengers per trip. Your GPU has allocated memory but it is not being used. 

You might be wondering, what is the optimal value to use? 

I usually go with 2-3x headroom of peak concurrent request. It means, if I am targeting 10rps, I'll test the system to handle upto 30rps. This way, with few extra resources you are ensuring availability of the model inference (in case of peak loads). Also, it leads to better GPU utilisation (due to better request batching and queuing).  


---

### 3. Batch Size Follows Concurrency

It took me a while to understand this. I always made this mistake to set `--max-num-batched-tokens` equal to the model's context length. So, if I am using llama-70B I would set `--max-num-batched-tokens=128000`. It caused the LLM inference to take longer. 

The pseudo code below explains better my concern:

```python
batch = []
batch_tokens = 0

for seq in active_sequences:
    # wait for batch to fill
    if batch_tokens + seq.tokens <= max_num_batched_tokens:
        batch.append(seq)
        batch_tokens += seq.tokens
    else:
        break  # Batch full, send to GPU

if batch_full or timeout:
    forward_pass(batch)
```

As you see, the vLLM server waits longer for the batch to get filled or the timeout is reached. When this happens, the GPU sits idle and the CPU is busy waiting/allocating tokens.

So, I prefer setting the `--max-num-batched-tokens` to (using values from above):

```
max_num_tokens = max_num_seqs * avg_tokens_per_seq
               = 58 * 500
               = 29000
```

Instead of context length, we should consider the average number of tokens in our input sequence.

---

### 4. Choose your Metrics

vLLM server exposes lot of metrics on its `/metrics` endpoint. To be precise I always monitor:

```bash
# vLLM metrics
curl http://localhost:8819/metrics

Key metrics:
- vllm:time_to_first_token_seconds (TTFT)
- vllm:time_per_output_token_seconds (TPOT)
- vllm:num_requests_running 
- vllm:num_requests_waiting 
- vllm:gpu_cache_usage_perc
- vllm:avg_generation_throughput_toks_per_s
```
Most of these metrics are self explanatory, so I'll focus on the first two:
- **TTFT**: If this is higher, means the prefill stage is causing slowness. This stage is compute bound. Try to reduce prompt size, check tokenizer, enable prefix caching or increase the batch size.  

- **TPOT**: If this is higher, means the decode stage is causing slowness. This stage is memory bound. Either use better attention mechanism (GQA >. MHA), smaller KV size or increase HBM (move to bigger hardware).   

For API server, I use [locust](https://locust.io/) and track the following metrics:

**Load test metrics:**
```python
# Track these over time:
- RPS (requests per second)
- p50, p95, p99 latency
- Failure rate
- Concurrent users supported
```

---

### 5: Plan Sequential LLM Calls

When your application makes multiple LLM calls per user request, you need MORE concurrent users to keep the GPU busy.

Lets see why:

*Single LLM call per request:*
```
User 1: [LLM call 300ms] → Finished
User 2: [LLM call 300ms] → Finished
User 3: [LLM call 300ms] → Finished

GPU batches Users 1, 2, 3 together → 300ms total
Throughput: Very High
GPU busy: 100%
```

*Multi-LLM calls per request (Our system):*
```
User 1: [Routing 300ms] → [Retrieval] → [Generation 300ms] → [Correction 300ms] → [Format 100ms]
Total: 1000ms per user

If only 1 user:
GPU timeline:
[Routing 300ms] [idle] [Generation 300ms] [idle] [Correction 300ms] [idle] [Format 100ms]

GPU busy: 700ms / 1000ms = 70% (30% idle between calls!)
```

This normally happens if you have are using a reasoning agent or a support agent, which needs to make multiple LLM calls to generate a response. 

In such situations, a better way to achieve higher GPU utilisation is by allowing more concurrent users:

*With concurrent users:*
```
User 1: [Routing] → ... → [Generation] → ...
User 2:           [Routing] → ... → [Generation] → ...
User 3:                     [Routing] → ... → [Generation] → ...

Now GPU can batch:
- User 1's Generation + User 2's Routing + User 3's initial call
- GPU busy: 95%+
```

The most common way to enable support for concurrent users it to fork more API workers. If you are using FastAPI, running the server behind gunicorn handles concurrency effortlessly.



## Summary

Besides the points I shared above, I also keep a closer look at GPU and CPU utilisation while testing the load. It gives me a different perspective about the server performance:

A higher CPU utilisation hints at:
- higher number of tokens
- higher time taken for tokenisation
- time taken by sampling operations (topk, top_p, minp etc)
- scheduling requests

A lower GPU utilisation hints at:
- suboptimal configuration
- lower load on GPU

I hope this post helped you to learn small but crucial things about deploying LLMs using vLLM. 

I am curious to hear what else do you do to make your LLM inference scalable.

---- 