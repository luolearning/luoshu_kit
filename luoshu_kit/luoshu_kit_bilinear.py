import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LuoshuFrame:
    def __init__(self, feature_shape, grid_policy=3, align_mode="nearest"):
        self.h, self.w = feature_shape[-2:]
        self.align_mode = align_mode

        if grid_policy == "auto":
            self.k = self._infer_k(max(self.h, self.w))
        elif isinstance(grid_policy, int):
            self.k = grid_policy
        else:
            raise ValueError(f"Unsupported grid_policy: {grid_policy}")

        if self.align_mode not in {"nearest", "bilinear"}:
            raise ValueError(f"Unsupported align_mode: {self.align_mode}")

        self.grid_size = 3 ** self.k
        self.luoshu_grid = self._gen_recursive_grid(self.k)
        self.seed_grid = self._gen_seed_grid()

    def _infer_k(self, size: int) -> int:
        return max(1, math.ceil(math.log(size, 3)))

    def _gen_seed_grid(self) -> torch.Tensor:
        return torch.tensor([
            [4.0, 9.0, 2.0],
            [3.0, 5.0, 7.0],
            [8.0, 1.0, 6.0],
        ])

    def _gen_recursive_grid(self, k: int) -> torch.Tensor:
        if k < 1:
            raise ValueError("k must be >= 1")
        if k == 1:
            return self._gen_seed_grid()

        grid = self._gen_seed_grid()
        for _ in range(2, k + 1):
            grid = grid.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)
        return grid

    def align(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        if feature_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N,C,H,W], got shape {tuple(feature_tensor.shape)}")

        if self.align_mode == "nearest":
            return F.interpolate(
                feature_tensor,
                size=(self.grid_size, self.grid_size),
                mode="nearest",
            )

        return F.interpolate(
            feature_tensor,
            size=(self.grid_size, self.grid_size),
            mode="bilinear",
            align_corners=False,
        )

    def grid_on(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        return self.luoshu_grid.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def seed_on(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        return self.seed_grid.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


class LuoshuBridge:
    def __init__(self, model: nn.Module, target_layer: str, frame: LuoshuFrame):
        self.model = model
        self.target_layer = target_layer
        self.frame = frame
        self.latest_features = None
        self.latest_stats = {}
        self._hook_handle = None

        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer '{target_layer}' not found in model.")

        def hook_fn(module, inputs, output):
            self.latest_features = output

        self._hook_handle = target_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _zero(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return torch.tensor(0.0, device=device)

    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=(-2, -1), keepdim=True)) / (
            x.std(dim=(-2, -1), keepdim=True) + 1e-6
        )

    def _calc_anchor_consistency(self, aligned_feat: torch.Tensor) -> torch.Tensor:
        if aligned_feat.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N,C,H,W], got {tuple(aligned_feat.shape)}")

        spatial = aligned_feat.abs().mean(dim=1)  # [N, G, G]
        spatial = self._normalize_map(spatial)

        seed = self.frame.seed_on(aligned_feat)   # [3, 3]
        seed = (seed - seed.mean()) / (seed.std() + 1e-6)

        G = self.frame.grid_size
        if G % 3 != 0:
            raise ValueError(f"Grid size must be divisible by 3, got {G}")

        block = G // 3

        # 把 [N, G, G] 切成 [N, 3, 3, block, block]
        patches = spatial.view(spatial.shape[0], 3, block, 3, block).permute(0, 1, 3, 2, 4)

        # 每个 block 内求平均，得到 [N, 3, 3]
        patch_means = patches.mean(dim=(-1, -2))
        patch_means = self._normalize_map(patch_means)

        diff = patch_means - seed.unsqueeze(0)
        return (diff ** 2).mean()

    def _calc_path_consistency(self, aligned_feat: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=aligned_feat.device, dtype=aligned_feat.dtype)

    def regularize(self, lambda_a=5.0, lambda_p=0.0):
        if self.latest_features is None:
            return self._zero()

        aligned_feat = self.frame.align(self.latest_features)
        anchor_loss = self._calc_anchor_consistency(aligned_feat)
        path_loss = self._calc_path_consistency(aligned_feat)

        self.latest_stats["anchor_loss"] = float(anchor_loss.detach().cpu())
        self.latest_stats["path_loss"] = float(path_loss.detach().cpu())
        self.latest_stats["grid_size"] = self.frame.grid_size
        self.latest_stats["k"] = self.frame.k
        self.latest_stats["feature_shape"] = tuple(self.latest_features.shape)
        self.latest_stats["align_mode"] = self.frame.align_mode

        return lambda_a * anchor_loss + lambda_p * path_loss

    def diagnostics(self):
        return dict(self.latest_stats)


def inject(model: nn.Module, layer_name: str, input_shape, grid_policy=3, align_mode="nearest", device=None):
    if device is None:
        device = next(model.parameters()).device

    cached = {"shape": None}

    def temp_hook(module, inputs, output):
        cached["shape"] = output.shape

    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    handle = target_module.register_forward_hook(temp_hook)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        _ = model(dummy)

    handle.remove()
    if was_training:
        model.train()

    if cached["shape"] is None:
        raise RuntimeError(f"Could not infer feature shape from layer '{layer_name}'.")

    frame = LuoshuFrame(
        feature_shape=cached["shape"],
        grid_policy=grid_policy,
        align_mode=align_mode,
    )
    bridge = LuoshuBridge(model=model, target_layer=layer_name, frame=frame)
    return bridge
