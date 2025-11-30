---
title: "Basics of Gradient Accumulation and Checkpointing to train LLMs"
date: 2025-11-29
tags:
- deep-learning
- pytorch
- llm
excerpt: "Basics of gradient accumulation and gradient checkpointing to train LLMs"
header:
  teaser: /assets/images/checkpoint_grad.png
  overlay_image: /assets/images/checkpoint_grad.png
  overlay_filter: 0.5
---

# Basics of Gradient Accumulation and Checkpointing to train LLMs

Fine training large language models (LLM) like GPT or BERT requires massive GPU memory. But what if you don't have access to a large hardware like H100, A100? Welcome **gradient accumulation** and **gradient checkpointing**—two essential techniques that let you train large models on limited consumer hardware.

In this post, we'll first understand how the memory is consumed when we fine tune a LLM. This way later we can acknowledge the importance of gradient accmulation and checkpointing.


## Explaining the problem

Let's say we want to train a neural network. Training a neural network requires storing:
1. **Model parameters** (weights): $\theta$
2. **Gradients**: $\nabla_\theta \mathcal{L}$
3. **Optimizer states**: momentum, variance (for Adam)
4. **Activations**: intermediate outputs from forward pass

So, for training a 7B parameter LLM in mixed precision (FP16), the memory required will be:

```
Parameters: 7B × 2 bytes = 14 GB
Gradients: 7B × 2 bytes = 14 GB
Optimizer states (Adam): 7B × 8 bytes = 56 GB
Activations: ~20-40 GB (depends on batch size)
──────────────────────────────────────────────
Total: ~104-124 GB
```

Let's breakdown some of those things. You might be wondering: why are optimizer stages so large? Why are gradients and activations requires separate memory? 

Let's dive in.


### Why Are Optimizer States So Large?

In general, optimizers plays a key role in updating the weights of a network. They make sure the gradient calculated is "decent" enough and is following the direction of minima.

**Basic weight update rule:**

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

Where:
- $\theta_t$ = parameters (weights) at step $t$
- $\eta$ = learning rate
- $\nabla_\theta \mathcal{L}$ = gradient of loss with respect to parameters

The problem here is, raw gradients are noisy and can vary wildly between batches. This is where optimizers like Adam come in.

The **Adam optimizer** (most commonly used for training LLMs) maintains **two moving averages** for each parameter:

1. **First moment (momentum)**: Exponential moving average of gradients - tracks the **direction** of gradient descent, by maintaining an average of past velocities.
2. **Second moment (variance)**: Exponential moving average of squared gradients - tracks the **scale** of gradients, adapting learning rate per parameter (larger for slowly-changing weights, smaller for rapidly-changing ones).

Below is a simplified code of how momentum and variance are updated during backward pass:

```python
import torch

# Initializing optimizer states for a single parameter (Wo)
param = torch.randn(100, 100)  # Example: a weight matrix
gradient = torch.randn(100, 100)  # Gradient from backprop

# Optimizer hyperparameters
beta1 = 0.9      # Momentum decay rate
beta2 = 0.999    # Variance decay rate
lr = 0.001       # Learning rate
epsilon = 1e-8   # Numerical stability

# Initialize states (stored in memory!)
m = torch.zeros_like(param)  # First moment (momentum) 
v = torch.zeros_like(param)  # Second moment (variance)
t = 0  # Time step

# Training loop
N = 100
for step in range(N):
    t += 1
    # Simulate getting gradient from backward pass
    # This comes from loss.backward()
    gradient = compute_gradient(param)  

    # Update momentum (exponential moving average of gradients)
    m = beta1 * m + (1 - beta1) * gradient
    # m stores: 90% of old momentum + 10% of new gradient

    # Update variance (exponential moving average of squared gradients)
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    # v stores: 99.9% of old variance + 0.1% of new squared gradient

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Update parameter using adaptive learning rate
    param = param - lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    #               ↑    ↑          ↑
    #               |    |          |
    #          learning  momentum   scale adjustment
    #           rate    (direction) (per-parameter)
```
As you can see, eventually it comes down to scaling `lr` using momentum and variance.

The inclusion of beta1 and beta2 makes sure the latest gradient shouldn't overpower the weight but instead gives higher weightage to gradients seen so far for stable learning.

Coming back to the problem above, for a 7B parameter model, we must store `m` and `v` for **all 7 billion parameters**!

- **Momentum** smooths out noisy gradients, preventing oscillations
- **Variance** adapts learning rate per parameter (big updates for stable params, small updates for volatile ones)
- Both must be stored in **FP32** for numerical stability (uses 4 bytes), even if model uses FP16
- This is why Adam uses **4× more memory** than just storing gradients!


## Gradients vs. Activations: What's the Difference?

Time to clarify another fundamental!

**Activations** are the outputs of each layer during the **forward pass**.
**Gradients** are the derivatives computed during the **backward pass**.

Think of it this way:
- **Activations flow forward** (input → output)
- **Gradients flow backward** (output → input)

Let's understand the forward pass using an example:

```python
import torch
import torch.nn as nn

# Define simple network
model = nn.Sequential(
    nn.Linear(2, 3),  # Layer1: 2 → 3
    nn.ReLU(),
    nn.Linear(3, 2),  # Layer2: 3 → 2
    nn.ReLU(),
    nn.Linear(2, 1),  # Output: 2 → 1
)

X = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])
```
When we run the forward pass `model.fit()`, it will store the activations at each layer, so in total there will be 5 intermediate tensors stored as activations.

**Why store activations?** We need them to compute gradients during backpropagation!

We should do a recap of what we've learnt so far about activations & gradients:

```
┌──────────────────┬──────────────────────────────────────────────┐
│ Activations      │ Gradients                                    │
├──────────────────┼──────────────────────────────────────────────┤
│ Computed during  │ Computed during BACKWARD pass                │
│ FORWARD pass     │                                              │
├──────────────────┼──────────────────────────────────────────────┤
│ One per layer    │ One per PARAMETER (weight/bias)              │
│ output           │                                              │
├──────────────────┼──────────────────────────────────────────────┤
│ Size depends on  │ Size = number of parameters                 │
│ batch size ×     │ (independent of batch size for storage,      │
│ hidden dims      │ but values depend on batch)                  │
├──────────────────┼──────────────────────────────────────────────┤
│ NEEDED to        │ NEEDED to update weights:                    │
│ compute          │ W_new = W_old - lr × gradient                │
│ gradients        │                                              │
├──────────────────┼──────────────────────────────────────────────┤
│ Can be           │ Must be kept (at least temporarily)          │
│ discarded after  │ for optimizer.step()                         │
│ backprop         │                                              │
│ (checkpointing!) │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

This tells us the following:
- Gradients scale with model size (number of parameters)
- Activations scale with batch size AND sequence length

For large transformers, activation memory dominates and grows with batch size. This is why:
- **Gradient checkpointing** targets activations (discard & recompute)
- **Gradient accumulation** keeps gradients small (accumulate across micro-batches)


## Gradient Accumulation: Simulate Larger Batches

Imagine you're moving bricks to build a wall. You can't carry 100 bricks at once, so you carry 10 bricks at a time, make 10 trips, then lay all 100 bricks together. Gradient accumulation works the same way.

Instead of processing a large batch at once, we:
1. Split the batch into **micro-batches**
2. Compute gradients for each micro-batch
3. **Accumulate** (sum) these gradients
4. Update weights after all micro-batches


**Standard training:**
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(X_{\text{batch}}, \theta_t)$$

**Gradient accumulation:**
$$\nabla_{\text{accum}} = \sum_{i=1}^{K} \nabla_\theta \mathcal{L}(X_{\text{micro-batch}_i}, \theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{K} \nabla_{\text{accum}}$$

Where $K$ is the number of accumulation steps.

To use gradient accumulation, your code should look something like below:

```python
import torch

model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Settings
batch_size = 64          # Effective batch size we want
micro_batch_size = 8     # What fits in GPU memory
accumulation_steps = batch_size // micro_batch_size  # = 8

optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Normalize loss to account for accumulation
    loss = loss / accumulation_steps

    # Backward pass (gradients accumulate)
    loss.backward()

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Gradient accumulation lets you use large effective batch sizes with small memory footprint, maintaining training quality.


## Gradient Checkpointing: Trading Compute for Memory

Think of gradient checkpointing as, instead of taking photos at every step of your hike (storing all activations), you only photograph major landmarks. When retracing your steps, you walk between landmarks to reconstruct the path.

During the **forward pass**, we:
1. Store activations only at **checkpoints** (not every layer)
2. Discard intermediate activations

During the **backward pass**, we:
1. Recompute discarded activations on-the-fly
2. Use them to calculate gradients

 

**Standard backpropagation:**
```
Forward:  Input → Layer1 → Layer2 → Layer3 → Output
          [Save] [Save]   [Save]   [Save]   [Save]

Backward: Compute gradients using saved activations
```

**With checkpointing:**
```
Forward:  Input → Layer1 → Layer2 → Layer3 → Output
          [Save] [Discard] [Save] [Discard] [Save]

Backward: Recompute Layer2 from Layer1
          Recompute Layer3 from Layer2
          Then compute gradients
```


Let's see a simple pseudocode implementation to get a better understanding:

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.ffn = FeedForward(d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        # Standard forward pass
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerWithCheckpointing(torch.nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Use gradient checkpointing for each layer
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

In real world, when you are fine tuning a model, you can use both the techniques together as shown below:

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model.gradient_checkpointing_enable()  # Reduce activation memory

training_args = TrainingArguments(
    per_device_train_batch_size=1,     # Micro-batch size
    gradient_accumulation_steps=32,     # Effective batch = 32
    fp16=True,                          # Mixed precision (2x memory savings)
    output_dir="./output",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## Summary

Gradient accumulation and checkpointing are not magic—they're clever trade-offs:

- **Gradient accumulation**: Trades time for effective batch size
- **Gradient checkpointing**: Trades compute for memory

Gradient Accumulation should be used when you want a large effective batch size but have limited hardware capacity. You'll want to use Gradient Checkpointing when the model has many layers (multiple transformer blocks) and activations dominate the memory usage.

Did you find this post useful? I am curious to hear from you in comments below.
