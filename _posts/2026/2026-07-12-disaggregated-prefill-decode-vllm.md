---
title: "Scaling LLM Inference with Disaggregated Prefill and Decode"
excerpt: "Disaggregated prefill/decode isn't an academic concept it's the fix that saved my LLM serving at scale for 40M users. Here's exactly how it works and why monolithic serving was the wrong default."
date: 2026-07-14
tags: [llm, inference, mlops]
header:
  teaser: /assets/images/disaggregated_prefill.png
  overlay_image: /assets/images/disaggregated_prefill.png
  overlay_filter: 0.5
---

Most teams running LLMs in production reach the same conclusion eventually. They profile their stack, see their expensive H100s sitting at 30–40% compute
utilization, and assume the fix is more GPUs. More GPUs. Better batching. Larger batch sizes. The usual levers.

They're pulling the wrong levers.

The real problem is that every LLM request contains **two completely different
jobs** and when you put them on the same GPU, they sabotage each other. One
job is starving the other. Until you separate them, you're leaving most of your hardware on the floor.

This is the story of how I built a disaggregated prefill/decode architecture
for a real-time chat suggestion system serving 40M monthly active users and what I learned pulling apart something most teams treat as a single monolith.



## Table of Contents
1. [The Problem: Two Jobs, One GPU](#the-problem-two-jobs-one-gpu)
2. [The Architecture](#the-architecture)
3. [Optimizations](#optimizations)
4. [What Actually Broke](#what-actually-broke-the-part-the-docs-dont-cover)
5. [Summary](#summary)



## The Problem: Two Jobs, One GPU

Let's follow one request through the system. A user sends a short message to a creator on the platform: *"hey, how was your day?"* The system needs to suggest a natural reply on behalf of the creator within 200ms.

Simple request. Should be fast. Right?

Here's what the actual token breakdown looked like:

| Component | Tokens |
|---|---|
| System prompt (creator persona, tone, platform rules) | ~1,800 |
| Conversation history (last 10 turns) | ~400 |
| Current user message | ~10 |
| **Total input** | **~2,210 tokens** |
| **Suggested reply** | **~30 tokens** |

The model reads 2,210 tokens to write 30. A 73:1 read-to-write ratio. And this is not an edge case, it's the median request. Chat context is always long.
Replies are always short.

This asymmetry is the entire story.

---

### Two Jobs, Zero Peace

To understand why this ratio matters, you need to understand what the GPU is actually doing at each stage.

**Prefill** (processing the 2,210 input tokens): The model runs a single forward pass over all input tokens simultaneously. This is a large dense matrix multiplication of thousands of tokens, millions of parameters, processed in parallel. The bottleneck is **compute**: FLOPs per second, tensor core utilization.

**Decode** (generating the 30 output tokens): The model generates one token at a time. Each step, it loads the entire model weight matrix from HBM to compute the next token's distribution. There's no parallelism across tokens, they're sequential by definition. The bottleneck is **memory bandwidth**: GB/s, not FLOPs.

> Prefill is compute-bound. Decode is memory-bandwidth-bound. These are
> different physics problems running on the same hardware.

On a single H100 SXM5:
- Peak compute: 989 TFLOPS (bf16)
- Peak memory bandwidth: 3.35 TB/s

A monolithic serving setup asks one GPU to be both a compute engine and a bandwidth engine simultaneously. When prefill dominates (my case: 2,210 input tokens, 30 output tokens), the long prefill phase blocks the GPU from starting decodes for new requests. TTFT (time to first token) climbs. Queues build silently.

Here's a simplified view of what the GPU timeline looks like in a monolithic setup when a heavy prefill request arrives alongside several decode-only requests:

<div style="background-color: #F9FAFB; border: 2px solid #D1D5DB; border-radius: 8px; padding: 10px; margin: 20px 0; position: relative;">
<div class="mermaid">
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#FFFFFF',
    'primaryTextColor': '#1F2937',
    'primaryBorderColor': '#9CA3AF',
    'lineColor': '#6B7280',
    'fontSize': '16px',
    'fontFamily': 'system-ui, -apple-system, sans-serif',
    'taskBkgColor': '#DBEAFE',
    'taskBorderColor': '#3B82F6',
    'activeTaskBkgColor': '#FEE2E2',
    'activeTaskBorderColor': '#EF4444',
    'activeTaskTextColor': '#1F2937',
    'gridColor': '#D1D5DB',
    'sectionBkgColor': '#F3F4F6',
    'altSectionBkgColor': '#E5E7EB',
    'titleColor': '#1F2937'
  },
  'gantt': {
    'useMaxWidth': true
  }
}}%%

gantt
    title Monolithic GPU Timeline — Prefill (red) blocks Decode (blue)
    dateFormat X
    axisFormat %s

    section GPU-0 (single node)
    Prefill req-A (2200 tokens)    :active, p1, 0, 40
    Decode req-B token 1           :d1, 40, 43
    Decode req-B token 2           :d2, 43, 46
    Decode req-B token 3           :d3, 46, 49
    Decode req-A token 1           :d4, 49, 52
    Prefill req-C (2100 tokens)    :active, p2, 52, 90
    Decode req-A token 2           :d5, 90, 93
</div>
<div style="text-align: center; color: #9CA3AF; font-size: 11px; margin-top: 8px;">© FloatingBytes | saraswatmks.github.io</div>
</div>

Req-B's decode is blocked waiting for req-C's prefill. TTFT for req-C's first token starts only after the prefill finishes. Everyone waits for everyone.

Before building anything, I profiled. The numbers made the decision obvious.

Using `nsys profile` on a saturated H100 at 1,000 req/s:

```python
# What I observed:
prefill_sm_utilization   = 0.89   # 89% — compute-saturated
decode_sm_utilization    = 0.12   # 12% — SM mostly idle, waiting on memory
decode_hbm_bandwidth_pct = 0.91   # 91% of peak HBM BW consumed during decode
```

Decode was using 12% of compute while consuming 91% of memory bandwidth. I was paying for a Ferrari to drive in first gear.

The fix was obvious: give prefill the compute, give decode the memory bandwidth. Stop compromising both.


## The Architecture

### Splitting Prefill from Decode

The disaggregated architecture separates the fleet into two specialized pools:
a **prefill pool** (compute-optimized, handles dense matrix ops) and a **decode
pool** (memory-bandwidth-optimized, handles one-token-at-a-time generation).

<div style="background-color: #F9FAFB; border: 2px solid #D1D5DB; border-radius: 8px; padding: 10px; margin: 20px 0; position: relative;">
<div class="mermaid">
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#FFFFFF',
    'primaryTextColor': '#1F2937',
    'primaryBorderColor': '#9CA3AF',
    'lineColor': '#6B7280',
    'fontSize': '16px',
    'fontFamily': 'system-ui, -apple-system, sans-serif'
  },
  'flowchart': {
    'nodeSpacing': 60,
    'rankSpacing': 100,
    'padding': 20,
    'useMaxWidth': true,
    'htmlLabels': true
  }
}}%%

flowchart LR
    GW[Gateway / Router]

    subgraph PrefillPool["Prefill Pool (4x H100)"]
        P0[H100-0]
        P1[H100-1]
        P2[H100-2]
        P3[H100-3]
    end

    subgraph DecodePool["Decode Pool (1x H100)"]
        D0[H100-4]
    end

    Client -->|request| GW
    GW -->|full prompt| PrefillPool
    PrefillPool -->|KV cache via NVLink| DecodePool
    DecodePool -->|generated tokens| GW
    GW -->|response| Client

    style GW fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style P0 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style P1 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style P2 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style P3 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style D0 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
</div>
<div style="text-align: center; color: #9CA3AF; font-size: 11px; margin-top: 8px;">© FloatingBytes | saraswatmks.github.io</div>
</div>

The request flow for the running example:

1. **Gateway** receives the 2,210-token chat request
2. **Prefill pool** runs the full forward pass, computes KV cache for all 2,210 tokens
3. **KV cache** transfers from the prefill node to the decode node
4. **Decode pool** generates tokens autoregressively, attending over the transferred KV cache
5. **Gateway** streams the 30-token reply back to the caller

The prefill pool never generates tokens. The decode pool never runs a full prefill. Each pool does one job.

In vLLM, this maps to running two separate server instances each configured
with `--kv-transfer-config`, using a connector such as `PyNcclConnector` (for intra-node NVLink) or `LMCacheConnectorV1` (the production-grade option backed
by the LMCache project, used in deployments at NVIDIA and Google Cloud). 

```python
# Prefill worker launch (simplified)
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3-70B-Instruct \
#     --tensor-parallel-size 4 \
#     --kv-transfer-config '{"kv_connector": "PyNcclConnector", "kv_role": "kv_producer"}' \
#     --port 8100

# Decode worker launch (simplified)
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3-70B-Instruct \
#     --tensor-parallel-size 1 \
#     --kv-transfer-config '{"kv_connector": "PyNcclConnector", "kv_role": "kv_consumer"}' \
#     --port 8200
```

### KV Cache Transfer: Moving Tensors Between Nodes

The most common question when people see this architecture: *"Doesn't the KV
cache transfer add latency?"*

It depends entirely on the transport layer. In my setup, all five H100s lived in a single DGX node connected by NVLink 4.0 at 900 GB/s. The KV transfer latency was under 2ms even for the largest requests.

```python
# KV cache size per token for a 70B LLaMA-style model
# Architecture: 80 layers, 8 KV heads (GQA), 128 head_dim, fp8 (1 byte)

num_layers       = 80
num_kv_heads     = 8
head_dim         = 128
bytes_per_element = 1   # fp8

kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
# = 163,840 bytes ≈ 160 KB per token

prompt_tokens = 2210
kv_transfer_size_mb = (kv_bytes_per_token * prompt_tokens) / (1024 ** 2)
print(f"KV transfer size: {kv_transfer_size_mb:.1f} MB")   # 345.0 MB

nvlink_bandwidth_gb_s = 900
transfer_latency_ms = (kv_transfer_size_mb / 1024) / nvlink_bandwidth_gb_s * 1000
print(f"Transfer latency: {transfer_latency_ms:.2f} ms")   # 0.37 ms
```

345MB in under 0.4ms. Negligible.

For cross-node setups (InfiniBand at 400 Gb/s ≈ 50 GB/s), the same transfer takes ~7ms — acceptable for most TTFT budgets. For Ethernet-only clusters, disaggregated prefill/decode is much harder to justify.

> The KV transfer cost is only worth it if your transport layer is fast enough.
> NVLink or InfiniBand: yes. Ethernet: profile first.

---

### Sizing the Fleet: How I Arrived at 4:1

The 4:1 ratio came from profiling the throughput ceiling of each pool
independently.

```python
req_per_sec        = 1000
avg_prefill_tokens = 2210
avg_decode_tokens  = 30

total_prefill_tokens_per_sec = req_per_sec * avg_prefill_tokens  # 2,210,000
total_decode_tokens_per_sec  = req_per_sec * avg_decode_tokens   #    30,000

# Empirical single-GPU throughput from vLLM metrics
prefill_throughput_per_gpu = 550_000  # tokens/sec (compute-saturated)
decode_throughput_per_gpu  = 35_000   # tokens/sec (memory-BW saturated)

import math
gpus_prefill = math.ceil(total_prefill_tokens_per_sec / prefill_throughput_per_gpu)
gpus_decode  = math.ceil(total_decode_tokens_per_sec  / decode_throughput_per_gpu)

print(f"Prefill GPUs needed: {gpus_prefill}")  # 5 → 4 with chunked prefill
print(f"Decode GPUs needed:  {gpus_decode}")   # 1
```

I also enabled **chunked prefill** which breaks large prompts into 512-token chunks. This flattened latency spikes from outlier prompts and reduced the prefill GPU requirement from 5 to 4 by improving utilization across requests.



## Optimizations

### Prefix Caching and Cache-Aware Routing

The 1,800-token system prompt is identical for every creator on the platform.
With prefix caching, the first request for creator X computes and stores that KV cache. Every subsequent request skips the system prompt prefill entirely.

In vLLM, prefix caching uses a **block-based hash table**. Memory is divided into 16-token blocks. A block is cached if its token content hash matches a previous request.

```python
block_size           = 16
system_prompt_tokens = 1800
conversation_tokens  = 400
new_turn_tokens      = 40   # only the latest turn is a miss

total_blocks      = (system_prompt_tokens + conversation_tokens) // block_size  # 137
new_turn_blocks   = new_turn_tokens // block_size                                # 2
cache_hit_blocks  = total_blocks - new_turn_blocks                               # 135

print(f"Cache hit rate: {cache_hit_blocks / total_blocks:.1%}")  # ~98.5%
```

A ~98% hit rate means 98% of requests skip the most expensive part of prefill.
But only if **the same creator's requests always route to the same prefill node** prefix cache lives in GPU HBM, not a shared store.

I built a stateful gateway that maps `creator_id → preferred_prefill_node`:

<div style="background-color: #F9FAFB; border: 2px solid #D1D5DB; border-radius: 8px; padding: 10px; margin: 20px 0; position: relative;">
<div class="mermaid">
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#FFFFFF',
    'primaryTextColor': '#1F2937',
    'primaryBorderColor': '#9CA3AF',
    'lineColor': '#6B7280',
    'fontSize': '16px',
    'fontFamily': 'system-ui, -apple-system, sans-serif'
  },
  'flowchart': {
    'nodeSpacing': 60,
    'rankSpacing': 100,
    'padding': 20,
    'useMaxWidth': true,
    'htmlLabels': true
  }
}}%%

flowchart LR
    R[Request<br/>creator_id=42] --> GW[Gateway]
    GW --> CM{Cache Map<br/>hit?}

    CM -->|Hit| P2[Prefill Node 2<br/>112 blocks cached]
    CM -->|Miss| LB[Load Balancer]
    LB --> P0[Prefill Node 0<br/>Cold prefill]

    P2 --> KV[KV Transfer]
    P0 --> KV
    KV --> D[Decode Node]
    D --> Resp[Response]

    style R fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style GW fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style CM fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style P2 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style LB fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style P0 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style KV fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style D fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style Resp fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
</div>
<div style="text-align: center; color: #9CA3AF; font-size: 11px; margin-top: 8px;">© FloatingBytes | saraswatmks.github.io</div>
</div>

When a prefill node's LRU cache starts evicting under memory pressure, the gateway detects it via vLLM's metrics endpoint and clears that creator's routing
entry. No sticky routing is permanent — just preferred.


## Where it failed slowly?

The happy path took a week to build. The edge cases took two months to stabilize. Here are the three failures that cost me the most time.


### Failure 1: The NCCL Deadlock

Throughput dropped to zero. No OOM. No crash. All pods healthy.
GPU utilization: 0% compute, 100% memory. The system was alive and doing nothing.

The prefill nodes had finished computing KV caches and were blocked on
`ncclSend()`, waiting for the decode node to call the matching `ncclRecv()`.
The decode node never called it, its block table was full from in-flight decode requests that hadn't finished yet.

NCCL collectives are synchronous barriers. Both sides must call the operation. If one side is stuck waiting for memory, the other hangs forever. No timeout by default.

```python
# Diagnosis: poll decode node's free block count before dispatching prefill
import requests
metrics = requests.get("http://decode-node:8200/metrics").text
# vllm:num_free_gpu_blocks — if 0, do not dispatch prefill
```

**Fix:** Backpressure signal from decode → gateway. If free blocks fall below 10% of total, the gateway queues the request locally instead of dispatching prefill. No NCCL send is ever initiated against a full decode node. Added ~5ms worst-case latency.


### Failure 2: The Hot Creator Problem

One prefill node at 100% utilization. Three others at 20%. P99 latency for one creator segment was 4x the rest. Cache hit rate was perfect, that was the problem.

Creator ID 8471 had 50,000 active fans. Every request routed to prefill-node-2. Exclusively. While nodes 0, 1, and 3 sat idle. The pure-hash routing had no load awareness at all.

```python
# Original — naive cache affinity, no load balancing
def route(creator_id, nodes):
    return nodes[hash(creator_id) % len(nodes)]  # creator_8471 → node_2, forever

# Fixed — cache affinity with load-aware spilling
def route(creator_id, nodes, metrics):
    preferred = nodes[hash(creator_id) % len(nodes)]
    load = metrics[preferred]["active_requests"] / metrics[preferred]["max_capacity"]

    if load > 0.85:
        # Accept cache miss, avoid queue buildup
        return min(nodes, key=lambda n: metrics[n]["active_requests"])
    return preferred
```

**Fix:** Above 85% utilization, spill to the least loaded node and accept one cold prefill hit. The 1,800-token cold prefill adds ~12ms to TTFT, far better than a 400ms queue. Cache miss rate from spilling at peak: ~8%. Mean TTFT improved because queue time was worse than the cold prefill cost.


### The edge case I still haven't fully solved
---

TP=4 on prefill means KV cache is sharded across 4 GPUs. Before transfer to the TP=1 decode node, the cache must be gathered onto one GPU first, an NCCL
all-gather across all 4 prefill GPUs. At peak load, these all-gathers compete
for NVLink bandwidth with ongoing prefill ops. I saw occasional 8–15ms KV
transfer spikes I couldn't fully explain or eliminate.

I think the cleaner fix could be TP=1 on prefill nodes (no gather needed), more replicas instead of tensor parallelism. A 70B fp8 model fits on one H100 (70GB model + 10GB KV). I haven't migrated yet. It's on the list.



## Summary

Building the disaggregated setup is quite powerful indeed but it comes with its own challenges. I wouldn't use disaggregated setup for usecases like code generation because the number or tokens generated by decode are a lot. Right? 

The uncomfortable truth is that most LLM serving guides treat the model as a
black box and tune serving parameters around it. The real gains come from
understanding what the model is actually doing on the hardware and building
your infrastructure around that, not around defaults.

Did you find this post useful? I am curious to hear from you in comments below.

