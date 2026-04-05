# LuoshuKit

Make neural networks learn **structured representations** in 3 lines.

LuoshuKit introduces **anchor–path constraints** that transform feature learning
from **search-based behavior into structured computation**.

---

## Quick Start

```python
from luoshu_kit import inject

bridge = inject(model, layer_name="layer1", input_shape=(1,3,32,32))
loss = ce_loss + bridge.regularize()
```

## What it does
Enforces structured feature maps
Encodes positional identity via path structure
Enables direct coordinate decoding (A2 regime)
Transforms learning from search → computation

