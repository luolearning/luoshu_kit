# LuoshuKit

Make your neural network learn structured representations in 3 lines.

Reduces anchor loss and enforces structured feature maps.

---

## ResNet Structural Signal

![ResNet](resnet_a0_a1_a2_structure.png)

ResNet layer2: A0 → A1 → A2

- A0: no structure  
- A1: anchor only  
- A2: path activated  

---

## Early Signal

![Early](early_signal.png)

Blockwise constraints enforce structured convergence.

---

## Usage

## Usage

LuoshuKit works as a plug-in. Simply inject it into an existing model.

Install:

```bash
git clone https://github.com/luolearning/luoshu_kit.git
cd luoshu_kit
pip install -e .
```
# Minimal example (python)
```
from luoshu_kit.luoshu_kit_a2_proto import inject

bridge = inject(model, layer_name="layer2")
