---
title: "Complete Guide to SFT and DPO Fine-tuning with Axolotl"
date: 2026-02-18
last_modified_at: 2026-02-18
layout: notebook
categories:
  - notebooks
tags:
  - deep-learning
  - llm
  - fine-tuning
excerpt: "Master the complete pipeline for fine-tuning Large Language Models using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) with Axolotl framework - from data preparation to production deployment"
header:
  teaser: /assets/images/notebooks/sft_dpo_notebook.png
  overlay_image: /assets/images/notebooks/sft_dpo_notebook.png
  overlay_filter: 0.5
classes: wide
# SEO optimization
seo_title: "Complete Guide to SFT + DPO Fine-tuning with Axolotl | Floating Bytes"
seo_description: "Master the complete pipeline for fine-tuning Large Language Models using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) with Axolotl framework"
keywords: "deep-learning, llm, fine-tuning, sft, dpo, rlhf, axolotl, lora, tutorial"
author: Manish Saraswat
---

# Overview

This notebook provides a comprehensive, production-ready guide to fine-tuning Large Language Models (LLMs) using **Supervised Fine-Tuning (SFT)** followed by **Direct Preference Optimization (DPO)** using the Axolotl framework.

### Why SFT → DPO?

- **SFT** teaches the model the task format and basic capabilities
- **DPO** refines the model to prefer better responses over worse ones
- This two-stage approach is more stable than direct DPO on base models
- DPO is simpler than RLHF (no reward model, no PPO complexity)

### Technical Architecture

```
Base Model (e.g., Llama-3.1-8B)
    ↓
SFT Training (Instruction Following)
    ↓
SFT Model Checkpoint
    ↓
DPO Training (Preference Alignment)
    ↓
Final Aligned Model
```

---

# Part 1: Environment Setup

First, we'll install Axolotl and its dependencies. Axolotl is a powerful framework that handles:
- Data preprocessing and formatting
- Model loading with quantization support
- Training with various optimizations (LoRA, Flash Attention, etc.)
- Evaluation and checkpointing

```python
# Install Axolotl and dependencies
# Note: This uses the latest Axolotl with Flash Attention 2 support
!pip install -q -U axolotl[flash-attn,deepspeed] accelerate transformers datasets bitsandbytes peft

# Import required libraries
import torch
import yaml
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Output:**
```
PyTorch version: 2.5.1
CUDA available: False
GPU: None
```

---

# Part 2: Understanding the Data Formats

### SFT Data Format

For SFT, we need instruction-response pairs. Modern formats:
- **ChatML format**: The current standard used by most models (GPT-4, Claude, Llama-3+)
- **Alpaca format**: Legacy single-turn instruction-input-output (mostly deprecated)
- **ShareGPT format**: Older multi-turn format (being phased out in favor of ChatML)

**ChatML is the modern standard** with better support and cleaner structure:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### DPO Data Format

DPO requires triplets:
- **Prompt**: The input question/instruction
- **Chosen**: The preferred response
- **Rejected**: The less preferred response

Let's load datasets from HuggingFace:

```python
# SFT Dataset: High-quality instruction-following data
# Using OpenOrca dataset - 1M GPT-4 and GPT-3.5 instruction-response pairs
sft_dataset = load_dataset("Open-Orca/OpenOrca", split="train[:5000]")  # Using 5k for demo

print("\n=== SFT Dataset Example ===")
print(sft_dataset[0])
print(f"\nSFT Dataset size: {len(sft_dataset):,}")

# DPO Dataset: Preference pairs
# Using Anthropic HH-RLHF dataset with human preferences
dpo_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

print("\n=== DPO Dataset Example ===")
print("Keys:", dpo_dataset.column_names)
print("\nChosen:", dpo_dataset[0]['chosen'][:200])
print("\nRejected:", dpo_dataset[0]['rejected'][:200])
print(f"\nDPO Dataset size: {len(dpo_dataset):,}")
```

**Output:**
```
=== SFT Dataset Example ===
{'id': 'niv.242684', 'system_prompt': '', 'question': "You will be given a definition of a task first...", 'response': '[\n  ["AFC Ajax (amateurs)", "has ground", "Sportpark De Toekomst"],\n  ["Ajax Youth Academy", "plays at", "Sportpark De Toekomst"]\n]'}

SFT Dataset size: 5,000

=== DPO Dataset Example ===
Keys: ['chosen', 'rejected']

Chosen: H: What are some cuss words in english?
A: Here's an incomplete list...

Rejected: H: What are some cuss words in english?
A: Here's an incomplete list...

DPO Dataset size: 5,000
```

---

# Part 3: Data Preprocessing

### Why Preprocessing Matters

1. **Format Standardization**: Different datasets have different structures
2. **Tokenization**: Convert text to tokens the model understands
3. **Special Tokens**: Add chat templates, EOS tokens, etc.
4. **Length Filtering**: Remove too-long sequences to save memory

Axolotl handles most of this, but we need to prepare our data correctly:

```python
# Convert OpenOrca to ChatML format (what Axolotl expects)
def convert_to_chatml_format(example):
    """
    Convert OpenOrca format to ChatML conversation format.

    ChatML format is the modern standard with a simple structure:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }

    Benefits of ChatML:
    - Native support in most modern LLMs (GPT-4, Claude, Llama-3+)
    - Better handling of multi-turn conversations
    - Standardized across the industry
    """
    messages = []

    # Add system message if present
    if example.get('system_prompt') and len(example['system_prompt'].strip()) > 0:
        messages.append({
            'role': 'system',
            'content': example['system_prompt']
        })

    # Add user question
    messages.append({
        'role': 'user',
        'content': example['question']
    })

    # Add assistant response
    messages.append({
        'role': 'assistant',
        'content': example['response']
    })

    return {'messages': messages}

# Apply conversion
sft_dataset_formatted = sft_dataset.map(
    convert_to_chatml_format,
    remove_columns=sft_dataset.column_names
)

print("\n=== Formatted SFT Example (ChatML) ===")
import json
print(json.dumps(sft_dataset_formatted[0], indent=2))
```

**Output:**
```json
{
  "messages": [
    {
      "content": "You will be given a definition of a task first...",
      "role": "user"
    },
    {
      "content": "[\n  [\"AFC Ajax (amateurs)\", \"has ground\", \"Sportpark De Toekomst\"],\n  [\"Ajax Youth Academy\", \"plays at\", \"Sportpark De Toekomst\"]\n]",
      "role": "assistant"
    }
  ]
}
```

```python
# Process DPO data - ensure it has required fields: prompt, chosen, rejected
def process_dpo_example(example):
    """
    Extract prompt, chosen, and rejected from Anthropic HH-RLHF format.

    The dataset uses special tokens:
    - \n\nHuman: marks user messages
    - \n\nAssistant: marks assistant messages
    """
    # Split on assistant marker to separate prompt from response
    chosen_parts = example['chosen'].split('\n\nAssistant:')
    rejected_parts = example['rejected'].split('\n\nAssistant:')

    return {
        'prompt': chosen_parts[0],  # Everything before first assistant response
        'chosen': chosen_parts[1].strip() if len(chosen_parts) > 1 else '',
        'rejected': rejected_parts[1].strip() if len(rejected_parts) > 1 else ''
    }

dpo_dataset_formatted = dpo_dataset.map(
    process_dpo_example,
    remove_columns=dpo_dataset.column_names
)

# Filter out any empty examples
dpo_dataset_formatted = dpo_dataset_formatted.filter(
    lambda x: len(x['chosen']) > 0 and len(x['rejected']) > 0
)

print("\n=== Formatted DPO Example ===")
print(f"Prompt: {dpo_dataset_formatted[0]['prompt'][:150]}...")
print(f"\nChosen: {dpo_dataset_formatted[0]['chosen'][:150]}...")
print(f"\nRejected: {dpo_dataset_formatted[0]['rejected'][:150]}...")
```

```python
# Save datasets to disk for Axolotl to load
os.makedirs('data', exist_ok=True)

sft_dataset_formatted.to_json('data/sft_data.jsonl')
dpo_dataset_formatted.to_json('data/dpo_data.jsonl')

print("✓ Datasets saved to data/ directory")
```

---

# Part 4: Supervised Fine-Tuning (SFT) Configuration

### Understanding the SFT Config

Key components:

1. **Base Model**: We'll use `meta-llama/Llama-3.2-1B` (small for demo, use 8B+ for production)
2. **LoRA**: Low-Rank Adaptation - efficient fine-tuning method
   - Only trains small adapter weights (~0.1-1% of model params)
   - Rank (r): Higher = more capacity but slower (8-64 typical)
   - Alpha: Scaling factor, usually 2x rank
3. **Training Params**:
   - Learning rate: 2e-5 is good for LoRA (higher than full fine-tuning)
   - Batch size: Trade-off between memory and training speed
   - Gradient accumulation: Simulate larger batches with limited memory
4. **Memory Optimization**:
   - Flash Attention 2: 2-4x faster attention
   - Gradient checkpointing: Trade compute for memory
   - BF16: Better than FP16 for training stability

```python
# SFT Configuration for Axolotl
sft_config = {
    # Base model configuration
    'base_model': 'meta-llama/Llama-3.2-1B',  # Use 'meta-llama/Llama-3.1-8B' for production
    'model_type': 'LlamaForCausalLM',
    'tokenizer_type': 'LlamaTokenizer',

    # Trust remote code (needed for some models)
    'trust_remote_code': True,

    # Data configuration - using modern ChatML format
    'datasets': [
        {
            'path': 'data/sft_data.jsonl',
            'type': 'chat_template',  # Use native chat template
            'chat_template': 'chatml',
            'field_messages': 'messages',  # Field containing the message list
            'message_field_role': 'role',  # Field for role (system/user/assistant)
            'message_field_content': 'content',  # Field for message content
        }
    ],

    # Data processing
    'dataset_prepared_path': 'last_run_prepared',
    'val_set_size': 0.05,  # 5% for validation
    'sequence_len': 2048,  # Max sequence length

    # LoRA configuration
    'adapter': 'lora',
    'lora_r': 32,  # Rank - controls adapter capacity
    'lora_alpha': 64,  # Scaling factor (typically 2x rank)
    'lora_dropout': 0.05,  # Regularization
    'lora_target_modules': [  # Which layers to adapt
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
    ],

    # Training hyperparameters
    'num_epochs': 1,  # Single epoch often sufficient with good data
    'micro_batch_size': 2,  # Per-device batch size
    'gradient_accumulation_steps': 4,  # Effective batch = 2 * 4 = 8
    'learning_rate': 2e-5,
    'lr_scheduler': 'cosine',  # Cosine annealing
    'warmup_steps': 100,  # LR warmup for stability

    # Optimizer
    'optimizer': 'adamw_torch',
    'weight_decay': 0.01,
    'adam_beta2': 0.95,  # Good for LLMs
    'adam_epsilon': 1e-8,

    # Memory optimization
    'flash_attention': True,  # Use Flash Attention 2
    'gradient_checkpointing': True,  # Save memory at cost of compute
    'bf16': True,  # BFloat16 training
    'fp16': False,  # Don't use FP16 if using BF16
    'tf32': True,  # TensorFloat32 for A100 GPUs

    # Logging and evaluation
    'logging_steps': 10,
    'eval_steps': 50,
    'save_steps': 100,
    'save_total_limit': 3,  # Keep only 3 checkpoints
    'output_dir': './sft_output',

    # Special tokens
    'special_tokens': {
        'pad_token': '<|pad|>',
        'eos_token': '<|end_of_text|>',
    },
}

# Save config
with open('sft_config.yml', 'w') as f:
    yaml.dump(sft_config, f, default_flow_style=False)

print("✓ SFT config saved to sft_config.yml")
print(f"\nKey settings:")
print(f"  Model: {sft_config['base_model']}")
print(f"  Data format: ChatML (modern standard)")
print(f"  LoRA rank: {sft_config['lora_r']}")
print(f"  Effective batch size: {sft_config['micro_batch_size'] * sft_config['gradient_accumulation_steps']}")
print(f"  Learning rate: {sft_config['learning_rate']}")
```

**Output:**
```
✓ SFT config saved to sft_config.yml

Key settings:
  Model: meta-llama/Llama-3.2-1B
  Data format: ChatML (modern standard)
  LoRA rank: 32
  Effective batch size: 8
  Learning rate: 2e-05
```

---

# Part 5: Running SFT Training

### Training Process

Axolotl will:
1. Load and tokenize the dataset
2. Initialize model with LoRA adapters
3. Train with gradient accumulation and checkpointing
4. Save best checkpoints
5. Merge LoRA weights back into base model (optional)

### Memory Requirements

For Llama-3.2-1B with LoRA:
- ~4-6 GB VRAM

For Llama-3.1-8B with LoRA:
- ~16-20 GB VRAM (single A100/A6000)
- Can use 4-bit quantization for ~10 GB VRAM

```python
# Run SFT training with Axolotl
# Note: In a real scenario, run this via command line for better logging:
# accelerate launch -m axolotl.cli.train sft_config.yml

!accelerate launch -m axolotl.cli.train sft_config.yml
```

### Understanding SFT Training Dynamics

**What to monitor:**

1. **Training Loss**: Should decrease smoothly
   - If it plateaus early: increase learning rate or model capacity
   - If it's unstable: decrease learning rate or increase warmup

2. **Validation Loss**: Should track training loss
   - If it diverges: you're overfitting, reduce epochs or add regularization

3. **Gradient Norm**: Should be stable (0.5-2.0 range)
   - If exploding: decrease LR or enable gradient clipping

4. **Learning Rate Schedule**: Cosine decay from peak to 0
   - Warmup prevents initial instability
   - Decay helps convergence

**Expected results after SFT:**
- Model understands instruction format
- Can generate coherent, on-topic responses
- May not always prefer "better" responses (that's what DPO fixes)

```python
# Load the SFT model for testing
from transformers import pipeline

# Merge LoRA weights into base model (optional, for easier deployment)
!python -m axolotl.cli.merge_lora sft_config.yml --lora_model_dir="./sft_output"

# Load merged model
sft_model_path = "./sft_output/merged"
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)

# Test the SFT model
test_prompt = "What is the capital of France?"
response = pipe(test_prompt)[0]['generated_text']

print("=== SFT Model Test ===")
print(f"Prompt: {test_prompt}")
print(f"Response: {response}")
```

---

# Part 6: Direct Preference Optimization (DPO) Configuration

### Understanding DPO

DPO is an elegant alternative to RLHF that directly optimizes the model to prefer chosen responses over rejected ones.

**Key insight**: Instead of training a reward model and using RL (PPO), DPO formulates preference learning as a classification problem.

**The DPO loss function:**

$$
L_{DPO} = -\log \sigma\left(\beta \left[\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x) - \log \pi_{ref}(y_w|x) + \log \pi_{ref}(y_l|x)\right]\right)
$$

Where:
- $\pi_\theta$: Current model being trained
- $\pi_{ref}$: Reference model (frozen SFT checkpoint)
- $y_w$: Chosen (winner) response
- $y_l$: Rejected (loser) response
- $\beta$: Temperature parameter (controls strength of optimization)

**Why this works:**
1. Increases probability of chosen responses
2. Decreases probability of rejected responses
3. KL penalty (via reference model) prevents mode collapse
4. No separate reward model needed!

### DPO Configuration Differences from SFT

1. **Base model**: Use SFT checkpoint (not base model)
2. **Reference model**: Keep frozen copy of SFT model
3. **Beta**: Controls optimization strength (0.1-0.5 typical)
4. **Learning rate**: Usually lower than SFT (1e-6 to 1e-5)
5. **Dataset**: Requires (prompt, chosen, rejected) triplets

```python
# DPO Configuration
dpo_config = {
    # Base model is now our SFT checkpoint
    'base_model': './sft_output/merged',
    'model_type': 'LlamaForCausalLM',
    'tokenizer_type': 'LlamaTokenizer',
    'trust_remote_code': True,

    # DPO-specific: reference model (frozen SFT model)
    'dpo_reference_model': './sft_output/merged',

    # Data configuration - DPO format
    'datasets': [
        {
            'path': 'data/dpo_data.jsonl',
            'type': 'dpo',  # DPO dataset type
        }
    ],

    'val_set_size': 0.05,
    'sequence_len': 2048,

    # LoRA configuration (same as SFT)
    'adapter': 'lora',
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.05,
    'lora_target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],

    # DPO-specific hyperparameters
    'dpo_beta': 0.1,  # KL penalty coefficient
    # Lower beta = more aggressive optimization, higher = more conservative
    # Start with 0.1, increase if model degrades

    # Training hyperparameters (more conservative than SFT)
    'num_epochs': 1,
    'micro_batch_size': 1,  # DPO needs 2x memory (stores chosen + rejected)
    'gradient_accumulation_steps': 8,  # Compensate for smaller batch
    'learning_rate': 5e-6,  # Lower than SFT to avoid forgetting
    'lr_scheduler': 'cosine',
    'warmup_steps': 50,

    # Optimizer
    'optimizer': 'adamw_torch',
    'weight_decay': 0.01,
    'adam_beta2': 0.95,
    'adam_epsilon': 1e-8,

    # Memory optimization (same as SFT)
    'flash_attention': True,
    'gradient_checkpointing': True,
    'bf16': True,
    'fp16': False,
    'tf32': True,

    # Logging and evaluation
    'logging_steps': 10,
    'eval_steps': 50,
    'save_steps': 100,
    'save_total_limit': 3,
    'output_dir': './dpo_output',

    # DPO evaluation metric
    'eval_metric': 'dpo_accuracy',  # % of times model prefers chosen over rejected
}

# Save config
with open('dpo_config.yml', 'w') as f:
    yaml.dump(dpo_config, f, default_flow_style=False)

print("✓ DPO config saved to dpo_config.yml")
print(f"\nKey settings:")
print(f"  Base (SFT) model: {dpo_config['base_model']}")
print(f"  DPO beta: {dpo_config['dpo_beta']}")
print(f"  Learning rate: {dpo_config['learning_rate']} (lower than SFT)")
print(f"  Effective batch size: {dpo_config['micro_batch_size'] * dpo_config['gradient_accumulation_steps']}")
```

**Output:**
```
✓ DPO config saved to dpo_config.yml

Key settings:
  Base (SFT) model: ./sft_output/merged
  DPO beta: 0.1
  Learning rate: 5e-06 (lower than SFT)
  Effective batch size: 8
```

---

# Part 7: Running DPO Training

### What Happens During DPO Training

1. **Forward pass on chosen**: Compute log probabilities for preferred response
2. **Forward pass on rejected**: Compute log probabilities for non-preferred response
3. **Reference model**: Compute log probs from frozen SFT model (for KL penalty)
4. **DPO loss**: Optimize to increase margin between chosen and rejected
5. **Backward pass**: Update only the policy model (reference stays frozen)

### Expected Memory Usage

DPO requires ~2x memory of SFT because:
- Must process both chosen and rejected sequences
- Must load reference model (can share weights with policy to save memory)

For Llama-3.2-1B:
- ~8-10 GB VRAM

For Llama-3.1-8B:
- ~24-32 GB VRAM (use 4-bit quant for 16GB)

```python
# Run DPO training
!accelerate launch -m axolotl.cli.train dpo_config.yml
```

### Understanding DPO Training Metrics

**Key metrics to monitor:**

1. **DPO Loss**: Should decrease steadily
   - Starts around 0.6-0.8 (random chance)
   - Good final value: 0.2-0.4
   - If it goes to 0: might be overfitting

2. **Reward Margin**: Difference in implied rewards between chosen and rejected
   - Should increase during training
   - Measures separation in model's preferences

3. **Accuracy**: How often model assigns higher probability to chosen response
   - Starts at ~50% (random)
   - Good final value: 70-85%
   - If 100%: likely overfitting

4. **KL Divergence**: Drift from reference model
   - Should be small (< 10)
   - If too high: model forgetting SFT capabilities
   - Increase beta if KL is too high

**Common issues:**

- **Reward hacking**: Model exploits preference function, degrades quality
  - Solution: Lower learning rate, increase beta, add more diverse data

- **Catastrophic forgetting**: Model loses SFT capabilities
  - Solution: Lower learning rate, stronger KL penalty (higher beta)

- **Slow convergence**: Training takes too long
  - Solution: Increase learning rate, clean data for stronger signal

```python
# Merge DPO LoRA weights
!python -m axolotl.cli.merge_lora dpo_config.yml --lora_model_dir="./dpo_output"

print("✓ DPO model merged and ready")
```

---

# Summary and Key Takeaways

### What We Accomplished

1. Set up Axolotl framework for efficient LLM training
2. Prepared datasets for both SFT and DPO
3. Fine-tuned a base model with SFT (instruction following)
4. Further aligned with DPO (preference learning)

### Best Practices

**Data Quality:**
- SFT: Use diverse, high-quality instruction-response pairs
- DPO: Ensure clear preference signal (chosen should be notably better)
- Filter out low-quality, contradictory, or harmful examples

**Training:**
- Start with smaller learning rates and scale up if needed
- Monitor validation metrics - don't just optimize training loss
- Use gradient checkpointing and Flash Attention for large models
- DPO should use lower LR than SFT to avoid catastrophic forgetting

**Memory Optimization:**
- LoRA for efficient fine-tuning (~10% of full fine-tuning memory)
- Flash Attention 2 for 2-4x speedup and memory savings
- Gradient checkpointing trades 30% compute for 50% memory savings
- BF16 for stability, better than FP16 for training

**Model Evaluation:**
- Use held-out validation set (5-10% of data)
- Monitor multiple metrics: loss, perplexity, task-specific metrics
- For DPO: track accuracy, reward margin, and KL divergence
- Test with qualitative examples throughout training

### Production Considerations

**Scaling to Larger Models:**
- Llama-3.1-8B: 16-24 GB VRAM with LoRA
- Llama-3.1-70B: Use multi-GPU with DeepSpeed ZeRO-3 or 4-bit quantization
- For very large models: Consider QLoRA (4-bit quantized base + LoRA)

**Deployment:**
- Merge LoRA weights into base model for simpler serving
- Use vLLM or TensorRT-LLM for optimized inference
- Quantize to INT8/INT4 for production if latency/cost critical

**Data Pipeline:**
- Invest in high-quality data curation
- Use GPT-4/Claude for synthetic data generation if needed
- Regularly audit for bias, toxicity, and factual errors
- Version control your datasets

### Final Notes

This notebook provided a complete, production-ready pipeline for SFT + DPO fine-tuning. The techniques here scale from 1B to 70B+ parameter models with appropriate hardware.

**Key Resources:**
- Axolotl Documentation: https://github.com/OpenAccess-AI-Collective/axolotl
- DPO Paper: https://arxiv.org/abs/2305.18290
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Flash Attention: https://arxiv.org/abs/2205.14135

**Next Steps:**
- Experiment with different hyperparameters (learning rate, LoRA rank, DPO beta)
- Try other alignment methods (PPO, REINFORCE, Constitutional AI)
- Implement evaluation harnesses (MT-Bench, AlpacaEval)
- Deploy with inference optimization (vLLM, quantization)
