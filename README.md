# LuoshuKit

**Make your neural network learn structured representations in 3 lines.**

**Reduces anchor loss and enforces structured feature maps.**
LuoshuKit is a lightweight plug-in that induces structured representations via blockwise regularization.

_No architectural changes required._

## Why LuoshuKit?

Neural networks typically learn unstructured feature maps.  
LuoshuKit adds a simple local constraint that makes representations more organized, consistent, and easier to analyze.

## Installation

Copy and run:

```bash
git clone https://github.com/luolearning/luoshu_kit.git
cd luoshu_kit
pip install -e .
```

## Usage


Minimal example:

from luoshu_kit.block_nearest import inject

bridge = inject(
    model,
    layer_name="features.2",
    input_shape=(4, 1, 28, 28),
    device=device,
)

loss = criterion(out, y) + bridge.regularize()

---


## Effect

![Luoshu Effect](luoshu_effect.png)

Blockwise constraints consistently induce structural convergence across interpolation schemes, while global regularization fails to do so.
