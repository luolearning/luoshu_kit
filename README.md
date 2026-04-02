# LuoshuKit

LuoshuKit is a lightweight toolkit for inducing structured computational representations in neural networks via blockwise regularization.

## Overview

Standard neural networks learn representations without explicit structure.  
LuoshuKit introduces a simple constraint that encourages feature maps to align with a structured grid (Luoshu), enabling deterministic computation over learned representations.

No architectural changes are required.

## Usage

Minimal example:

```python
from luoshu_kit.block_nearest import inject

bridge = inject(
    model,
    layer_name="features.2",
    input_shape=(4, 1, 28, 28),
    device=device,
)

loss = criterion(out, y) + bridge.regularize()
```

## Installation

```bash
git clone https://github.com/luolearning/luoshu_kit.git
cd luoshu_kit
```
