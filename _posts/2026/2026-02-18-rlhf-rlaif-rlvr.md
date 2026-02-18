---
title: "RLHF vs RLAIF vs RLVR: The Three Ways to Teach AI Models"
date: 2026-02-18
tags:
- llm
excerpt: "Understanding the basics of RLHF vs RLAIF vs RLVR for AI feedback comparison"
header:
  teaser: /assets/images/rlhf_rlaif_rlvr_blog.png
  overlay_image: /assets/images/rlhf_rlaif_rlvr_blog.png
  overlay_filter: 0.5
---


# RLHF vs RLAIF vs RLVR: The Three Ways to Teach AI Models


## Table of Contents
- [1. The Problem (Why Pretrained Models Are Like Smart Toddlers)](#1-the-problem-why-pretrained-models-are-like-smart-toddlers)
- [2. RLHF: Teaching with Humans (The Original Method)](#2-rlhf-teaching-with-humans-the-original-method)
- [3. RLAIF: Teaching with AI Judges (The Scalable Rebellion)](#3-rlaif-teaching-with-ai-judges-the-scalable-rebellion)
- [4. RLVR: Teaching with Math (The Verifiable Dream)](#4-rlvr-teaching-with-math-the-verifiable-dream)
- [5. When to Choose Which Method](#5-when-to-choose-which-method)
- [Summary](#summary)


So you've trained a massive language model on the entire internet. Congratulations! It can write poetry, answer trivia, explain quantum physics, and also tell you how to hotwire a car or synthesize new colors. But, its not enough.

Your model is incredibly smart but has the moral compass of a very educated toddler. It learned to predict text, not to distinguish helpful from harmful, truthful from deceptive, or useful from toxic. And that's a problem.

## 1. The Problem (Why Pretrained Models Are Like Smart Toddlers)

Let's say you ask GPT-4 (before any safety training): How do I make my code run faster? It might give you brilliant optimization tips. Or it might tell you to "delete all your tests because they slow down execution." Both are *technically* text that could follow your prompt on the internet. Pretraining doesn't distinguish quality.

This happens because *pretraining is basically next-token* prediction on steroids. 

The model learns "what word comes next" across billions of documents, which teaches it grammar, facts, reasoning patterns, and unfortunately also conspiracy theories, offensive content, and deeply unhelpful responses. It has no built-in notion of "this answer is actually good" versus "this answer is technically possible but terrible."

The solution? We need to teach the model human preferences, what responses we actually want. Let's look at different possible ways and understand the tradeoffs between their cost, scale, and accuracy.

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
    A[Pretrained Model<br/>next token prediction] --> B{Alignment Training}
    B --> C[RLHF<br/>humans rate outputs]
    B --> D[RLAIF<br/>AI judges outputs]
    B --> E[RLVR<br/>math verifies outputs]
    C --> F[Aligned Model<br/>helpful & safe]
    D --> F
    E --> F

    style A fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style B fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style C fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style D fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style E fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style F fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
</div>

<div style="text-align: center; color: #9CA3AF; font-size: 11px; margin-top: 8px;">© FloatingBytes | saraswatmks.github.io</div>

</div>

So how do we actually teach preferences? Let's start with the method that started it all.

## 2. RLHF: Teaching with Humans (The Original Method)

RLHF stands for *Reinforcement Learning from Human Feedback*, and it's the approach that powered InstructGPT (which became ChatGPT) and basically created the modern aligned language model category. Here's how it works:

- First, you gather human annotators—actual people sitting at computers. You show them pairs of model outputs for the same prompt. "Which response is better?" they click. You do this tens of thousands of times across diverse prompts. Those comparisons become your training data.
- But you can't directly train on "Susan preferred output A over output B." You need something differentiable. So you train a reward model, which is itself a neural network that learns to predict which outputs humans prefer. This reward model becomes your proxy for human judgment.

The math here is actually pretty elegant. You're modeling human preferences using the **Bradley-Terry** model, which says the probability that output A beats output B depends on their relative rewards:

$$P(A > B) = \frac{e^{r(A)}}{e^{r(A)} + e^{r(B)}} = \sigma(r(A) - r(B))$$

Where $r(A)$ is the reward model's score for output A, and $\sigma$ is the sigmoid function. Basically: bigger reward difference means more confident preference.

Once you have this reward model, you use it to fine-tune your language model using reinforcement learning, specifically PPO (Proximal Policy Optimization). 

The model generates outputs, the reward model scores them, and PPO adjusts the model's parameters to produce higher-scoring outputs. You also add a KL divergence penalty to prevent the model from drifting too far from its original pretrained behavior, because if you optimize too hard for the reward, you get reward hacking (the model finds weird exploits that technically score high but are nonsense).

Fine, too much of talking. Lets see how we can do this using pytorch:

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # Pretrained LM
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        # Take last token's hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.value_head(last_hidden)
        return reward

# Training loop for reward model
def train_reward_model(model, pairs_dataset):
    """pairs_dataset contains (chosen_text, rejected_text) tuples"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for chosen_ids, rejected_ids in pairs_dataset:
        r_chosen = model(chosen_ids, attention_mask=None)
        r_rejected = model(rejected_ids, attention_mask=None)

        # Bradley-Terry loss: chosen should have higher reward
        loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
```

So that's the RLHF pipeline—humans label comparisons, you train a reward model, then you fine-tune the language model using RL to maximize that reward. OpenAI used this for ChatGPT and reported massive improvements in helpfulness and safety.

But here's the problem, human annotation is incredibly expensive and slow. Training a model like ChatGPT's reward model might cost $500K-1M just in labeling. 

And what if you want to improve the model every week? Or support 50 languages? You'd need an army of annotators. Which brings us to the rebellious idea.

## 3. RLAIF: Teaching with AI Judges (The Scalable Rebellion)

What if you just asked a different AI to do it? Like, you already have GPT-5 or Claude sitting around. Why not use them as judges?

This is RLAIF: **Reinforcement Learning from AI Feedback**. The core insight is that modern language models are actually pretty good at evaluating outputs if you give them clear criteria. So instead of showing comparison pairs to humans, you show them to an AI judge with a detailed prompt.

The AI judge prompt might say: "You are an expert evaluator. Compare these two responses for helpfulness, harmlessness, and accuracy. Explain your reasoning, then pick the better response." 

The AI writes out its reasoning (which helps reduce inconsistency) and makes a choice. You collect tens of thousands of these AI judgments instead of human ones, then proceed with the exact same reward modeling and RL fine-tuning pipeline as RLHF.

Anthropic pioneered this approach with **Constitutional AI**, where they use AI feedback to teach models their "constitution"—a set of principles about being helpful, honest, and harmless. They found that AI generated feedback could match or exceed human feedback quality for many tasks, while being 10-100x faster and cheaper to generate.

Lets see how RLHAIF can be done using code:

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def get_ai_judgment(prompt, output_a, output_b):
    """Ask AI judge which output is better"""
    judge_prompt = f"""You are an expert evaluator. Compare these two AI responses.

User prompt: {prompt}

Response A: {output_a}

Response B: {output_b}

Which response is better in terms of helpfulness, accuracy, and safety?
Explain your reasoning, then output 'A' or 'B'."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.3  # Lower temperature for more consistent judgments
    )

    judgment_text = response.choices[0].message.content
    # Parse the final A or B choice from the response
    return "A" if judgment_text.strip().endswith("A") else "B"

# Generate comparison dataset using AI judge
comparison_data = []
for prompt, output_a, output_b in model_outputs:
    winner = get_ai_judgment(prompt, output_a, output_b)
    if winner == "A":
        comparison_data.append((output_a, output_b))  # (chosen, rejected)
    else:
        comparison_data.append((output_b, output_a))
```

The key question is: how good are AI judges compared to humans? The answer turns out to be "surprisingly good, but with caveats." 

Research shows AI judges agree with humans 70-85% of the time on straightforward tasks (helpfulness, clarity, following instructions). They struggle more with subjective preferences, cultural nuance, and subtle safety issues. 

But for many applications, that's good enough—especially since you can use multiple AI judges and ensemble their votes to improve reliability.

The cost difference is staggering. Where RLHF might cost $500K for 50,000 labels, RLAIF might cost $5K in API calls. You can iterate weekly instead of quarterly. You can support 50 languages without hiring annotators for each one. But you're trading away some accuracy and introducing the risk that your judge model's biases become baked into your aligned model.

Now, lets look at the third way, what if you didn't need judges at all?

## 4. RLVR: Teaching with Math (The Verifiable Dream)

RLVR stands for **Reinforcement Learning from Verifiable Rewards**, and it's the most elegant approach of the three when it works. The idea is simple: 

> For certain tasks, you can automatically verify whether an answer is correct without any human or AI judgment. You just run the code, check the math, or compare against a known solution.

Think about coding tasks. If the model generates Python code to solve a problem, you don't need a human to rate it. You run the code on test cases. If it passes all tests, reward equals 1. If it fails, reward equals 0. No ambiguity, no annotation cost, perfect ground truth.

Same thing works for math problems (check if the final answer is correct), factual questions with known answers (lookup in a knowledge base), code translation (does the translated code produce identical outputs?), and various reasoning tasks where you can verify the chain of logic.

The RLVR training pipeline is actually simpler than RLHF or RLAIF. You don't need to train a reward model at all—you have a programmatic reward function. You just generate model outputs, run automatic verification, assign rewards, and use RL (typically PPO) to fine-tune the model toward higher rewards. The KL penalty still applies to prevent the model from degenerating.

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
    A[Prompt] --> B[Model generates code]
    B --> C[Run test cases]
    C --> D{All tests pass?}
    D -->|Yes| E[Reward = 1]
    D -->|No| F[Reward = 0]
    E --> G[RL update]
    F --> G
    G --> A

    style A fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style B fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style C fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style D fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style E fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style F fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style G fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
</div>

<div style="text-align: center; color: #9CA3AF; font-size: 11px; margin-top: 8px;">© FloatingBytes | saraswatmks.github.io</div>

</div>

**What this does**: Automatically verifies code correctness using test cases instead of human/AI judgment.

```python
def verify_code_solution(generated_code, test_cases):
    """
    Automatically verify if generated code solves the problem.
    Returns reward in [0, 1] based on test case pass rate.
    """
    passed = 0
    total = len(test_cases)

    for inputs, expected_output in test_cases:
        try:
            # Execute generated code in isolated environment
            namespace = {}
            exec(generated_code, namespace)

            # Assume code defines a 'solution' function
            actual_output = namespace['solution'](*inputs)

            if actual_output == expected_output:
                passed += 1
        except Exception:
            # Code crashed or has errors
            pass

    return passed / total  # Reward between 0 and 1

# RLVR training doesn't need a reward model
# Just use verify_code_solution as the reward function during RL
def rlvr_training_step(model, prompt, test_cases):
    generated_code = model.generate(prompt)
    reward = verify_code_solution(generated_code, test_cases)

    # PPO update using this reward (simplified)
    # model.update_policy(reward, generated_code, prompt)
    return reward
```

RLVR provides perfect ground truth for verifiable tasks, eliminating human and AI biases entirely.

OpenAI used something similar to this for training their code models, models that solve coding problems get strong reward signal from automatic test verification. 

But the catch is obvious: this only works when you *have* automatic verification. You can't verify when subjectivity exists like: "is this response empathetic?" or "does this answer sound natural?" or "is this creative writing compelling?" For most open-ended language tasks, RLVR simply doesn't apply. That's why it's powerful but narrow.

## 5. When to Choose Which Method

Here's a decision tree diagram which I could think of, can give a clear guidance on how to choose the optimal method:

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

flowchart TD
    Start[Which alignment method?] --> Q1{Is task<br/>verifiable?}

    Q1 -->|Yes<br/>Code, math,<br/>test cases| RLVR[Use RLVR<br/>Perfect accuracy<br/>Cost: 1K<br/>Time: 3 days]

    Q1 -->|No<br/>Subjective,<br/>open-ended| Q2{Budget and<br/>timeline?}

    Q2 -->|500K-1M<br/>2-3 months| Q3{Quality<br/>critical?}

    Q3 -->|Yes<br/>Safety-critical<br/>consumer product| RLHF[Use RLHF<br/>85-95% quality<br/>Gold standard<br/>Slow but best]

    Q3 -->|No<br/>Internal tools<br/>prototypes| RLAIF1[Use RLAIF<br/>70-85% quality<br/>Fast iteration]

    Q2 -->|5K-50K<br/>1-2 weeks| Q4{Need multilingual<br/>or rapid iteration?}

    Q4 -->|Yes| RLAIF2[Use RLAIF<br/>10-100x cheaper<br/>Support 50+ langs]

    Q4 -->|No but need<br/>high quality| Hybrid[Hybrid Approach<br/>RLHF for core<br/>RLAIF for iteration]

    style Start fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style Q1 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style Q2 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style Q3 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style Q4 fill:#FFFFFF,stroke:#9CA3AF,stroke-width:2px,color:#1F2937
    style RLVR fill:#D1FAE5,stroke:#10B981,stroke-width:3px,color:#065F46
    style RLHF fill:#DBEAFE,stroke:#3B82F6,stroke-width:3px,color:#1E40AF
    style RLAIF1 fill:#FEF3C7,stroke:#F59E0B,stroke-width:3px,color:#92400E
    style RLAIF2 fill:#FEF3C7,stroke:#F59E0B,stroke-width:3px,color:#92400E
    style Hybrid fill:#E9D5FF,stroke:#A855F7,stroke-width:3px,color:#6B21A8
</div>

<div style="text-align: center; color: #9CA3AF; font-size: 11px; margin-top: 8px;">© FloatingBytes | saraswatmks.github.io</div>

</div>

You can also combine methods. Many organizations use RLHF for the initial alignment on core capabilities and safety, then use RLAIF for rapid iteration on specific skills or style preferences.

The cost comparison is dramatic. Training a reward model with RLHF might cost `$500K` and take two months. RLAIF might cost `$5K` and take two weeks. RLVR might cost `$1K` in compute and take three days. But RLHF gets you 85-95% human agreement, RLAIF gets you 70-85%, and RLVR gives you 100% accuracy but only works on 10% of tasks.

## Summary

All three of these methods are ultimately training the model to optimize for *some proxy* of what we actually want. 

We learnt that: 
- RLHF uses expensive human feedback to train reward models and align language models, highest quality but slow and costly. 
- RLAIF replaces humans with AI judges 10-100x cheaper and faster but slightly lower quality. 
- RLVR uses automatic verification for math and code, perfect accuracy when applicable but only works on objectively verifiable tasks. 

Funny part is, none of them actually optimize for "help the human achieve their goals" or "make the world better." We've gotten incredibly sophisticated at aligning models to our feedback signals, but the question of whether those feedback signals capture what we *really* want remains deeply unsolved.

Still, they're the best we've got right now. And understanding their tradeoffs helps you build better systems, spend your budget wisely, and at least be aware of the limitations. Which is more than most people can say.
