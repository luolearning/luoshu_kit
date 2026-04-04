# LuoshuKit

**Make your neural network learn structured representations in 3 lines.**

LuoshuKit is a lightweight plug-in that induces structured representations via blockwise regularization.

_No architectural changes required._

---

## Why LuoshuKit?

Neural networks typically learn unstructured feature maps.  
LuoshuKit adds a simple local constraint that makes representations more organized, consistent, and easier to analyze.

---

## Installation

Copy and run:

```python
loss = criterion(out, y) + bridge.regularize()
