# LuoshuKit

**Make your neural network learn structured representations in 3 lines.**

**Reduces anchor loss and enforces structured feature maps.**

## Effect

![Luoshu Effect](luoshu_effect.png)

Blockwise constraints consistently reduce anchor loss, leading to more structured and stable representations.

## Why LuoshuKit?

LuoshuKit induces structured representations via simple blockwise regularization—no architectural changes required.

It transforms unstructured feature maps into more organized and consistent forms, making them easier to analyze.

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
