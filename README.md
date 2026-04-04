# LuoshuKit

**Make your neural network learn structured representations in 3 lines.**

**Reduces anchor loss and enforces structured feature maps.**

## Effect

![Luoshu Effect](luoshu_effect.png)

Blockwise constraints consistently induce structural convergence across interpolation schemes, while global regularization fails to do so.
Lower anchor loss indicates more structured and stable feature representations.


## Why LuoshuKit?

LuoshuKit is a lightweight plug-in that induces structured representations via blockwise regularization, requiring no architectural changes. 

Neural networks typically learn unstructured feature maps; LuoshuKit introduces a simple local constraint that makes them more organized, consistent, and easier to analyze.


## Installation

Copy and run:

```bash
git clone https://github.com/luolearning/luoshu_kit.git
cd luoshu_kit
pip install -e .
```

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
