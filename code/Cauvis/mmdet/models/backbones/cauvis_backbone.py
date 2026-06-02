import torch
from .reins_utils import set_requires_grad, set_train
from mmdet.registry import MODELS
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from .dino_v2 import DinoVisionTransformer

@MODELS.register_module()
class CauvisDINOv2(DinoVisionTransformer):
    def __init__(self,
                 cauvis_config=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.cauvis = MODELS.build(cauvis_config)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.cauvis.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["cauvis"])
        set_train(self, ["cauvis"])
