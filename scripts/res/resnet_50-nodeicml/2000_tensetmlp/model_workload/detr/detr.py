# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone, FrozenBatchNorm2d
from .ops import NestedTensor, nested_tensor_from_tensor_list, unused
from .transformer import build_transformer



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    def __init__(self, backbone, position_embedding, hidden_dim, num_classes, num_queries, aux_loss=False, pretrained=False):
        super().__init__()
        self.backbone       = build_backbone(backbone, position_embedding, hidden_dim, pretrained=pretrained)
        self.input_proj     = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        
        self.transformer    = build_transformer(hidden_dim=hidden_dim, pre_norm=False)
        hidden_dim          = self.transformer.d_model
        self.class_embed    = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed     = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed    = nn.Embedding(num_queries, hidden_dim)
        
        self.num_queries    = num_queries
        self.aux_loss       = aux_loss

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return outputs_coord
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, FrozenBatchNorm2d):
                m.eval()


def detr():
    backbone = "resnet50"
    num_classes = 10
    model = DETR(backbone, 'sine', 256, num_classes, 100, pretrained=False)
    return model

if __name__ == "__main__":
    model = detr()
    x = torch.randn((1,3,256,256))
    out = model(x)
