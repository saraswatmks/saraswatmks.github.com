# Jupyter Notebook Style Guide

## Overview

This site uses a custom Jupyter notebook styling system that gives your markdown-based notebooks an authentic Jupyter look while maintaining perfect SEO and integration with Jekyll.

## How to Use

### 1. Add `layout: notebook` to Frontmatter

In your notebook markdown file, add `layout: notebook` to the frontmatter:

```yaml
---
title: "Your Notebook Title"
date: 2026-02-18
layout: notebook  # ← Add this line
categories:
  - notebooks
tags:
  - your-tags
---
```

### 2. Write Your Content in Markdown

The notebook layout automatically styles your markdown to look like Jupyter:

#### Code Cells (Input)

Use standard markdown code blocks with language specification:

````markdown
```python
# Your Python code here
import torch
print("Hello World")
```
````

This will render with:
- "In [ ]:" label on the left
- Jupyter-style code cell background
- Proper syntax highlighting

#### Output Cells

For output, use either:

**Option A: Blockquotes** (Recommended for simple output)
```markdown
> PyTorch version: 2.5.1
> CUDA available: True
```

**Option B: Bold text** (For inline output)
```markdown
**Output:**
```
PyTorch version: 2.5.1
CUDA available: True
```
```

Both render with:
- "Out[ ]:" label on the left
- White background (like Jupyter output cells)
- Monospace font

#### Markdown Cells

Regular markdown text, headers, lists, tables, etc. render as standard Jupyter markdown cells.

### 3. Example Notebook Structure

```markdown
---
title: "My Notebook"
layout: notebook
categories:
  - notebooks
---

# Introduction

This is a markdown cell explaining the notebook.

## Setup

```python
import torch
import numpy as np

print(f"PyTorch: {torch.__version__}")
```

**Output:**
```
PyTorch: 2.5.1
```

## Analysis

Here's a table of results:

| Model | Accuracy |
|-------|----------|
| GPT-4 | 95.2%    |
| Claude| 94.8%    |

```python
# Train model
model.train()
```

> Epoch 1/10: Loss = 0.342
> Epoch 2/10: Loss = 0.298
```

## Features

### Automatic Styling

- ✅ **Code cells**: Styled with "In [ ]:" labels
- ✅ **Output cells**: Styled with "Out[ ]:" labels
- ✅ **Tables**: Enhanced with hover effects
- ✅ **Images**: Centered with shadows
- ✅ **Lists**: Proper spacing and indentation
- ✅ **Math equations**: Supports MathJax/KaTeX
- ✅ **Mobile responsive**: "In/Out" labels hidden on mobile

### Supported Languages

The Jupyter styling works with:
- `python`
- `bash` / `shell`
- `json`
- `yaml`
- Any other language supported by Rouge/Pygments

### Desktop vs Mobile

**Desktop (>768px):**
- Shows "In [ ]:" and "Out[ ]:" labels on the left
- 80px left margin for labels
- Full Jupyter experience

**Mobile (≤768px):**
- Labels hidden for better readability
- Full-width code cells
- Optimized for small screens

## Customization

The Jupyter styling is defined in `_sass/jupyter-notebook.scss`. You can customize:

- Colors (change `#303f9f` for "In" label color, `#d84315` for "Out" label color)
- Fonts (change `'Monaco', 'Menlo', ...` to your preferred monospace font)
- Spacing (adjust margins and padding)
- Cell borders and shadows

## Tips

1. **For clean output blocks**, use blockquotes:
   ```markdown
   > Line 1 of output
   > Line 2 of output
   ```

2. **For multi-line code output**, use code blocks inside blockquotes:
   ```markdown
   > ```
   > Multi-line
   > code output
   > ```
   ```

3. **For images**, just use regular markdown:
   ```markdown
   ![Description](path/to/image.png)
   ```
   They'll automatically be centered with a shadow.

4. **For tables**, use standard markdown tables - they'll get Jupyter styling automatically.

## Converting Jupyter Notebooks

To convert `.ipynb` files to markdown:

```bash
jupyter nbconvert --to markdown your_notebook.ipynb
```

Then:
1. Move the `.md` file to `_posts/YEAR/`
2. Add the frontmatter with `layout: notebook`
3. Adjust any image paths
4. Wrap output in blockquotes or bold text sections

## Examples

See these notebooks for examples:
- [Complete Guide to SFT and DPO Fine-tuning with Axolotl](_posts/2026/2026-02-18-complete-guide-sft-dpo-finetuning-axolotl.md)

## Troubleshooting

**Q: Code cells don't have "In [ ]:" labels**
- Make sure you're using `layout: notebook` in frontmatter
- Ensure code blocks have language specification (```python, not just ```)

**Q: Output doesn't have "Out[ ]:" labels**
- Wrap output in blockquotes (`> text`) or bold sections

**Q: Styling not working**
- Make sure `_sass/jupyter-notebook.scss` is imported in `assets/css/main.scss`
- Clear your Jekyll cache: `bundle exec jekyll clean`
- Rebuild: `bundle exec jekyll serve`

**Q: Labels showing on mobile**
- This is controlled by media queries - check `_sass/jupyter-notebook.scss`
