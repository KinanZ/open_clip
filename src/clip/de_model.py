import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel
import torch.nn.functional as F
import numpy as np


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisionEncoder(nn.Module):
    def __init__(self, resnet_model: str, pretrained: bool, d_out: int, freeze_bb: bool) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.resnet_model = resnet_model
        self.freeze_bb = freeze_bb
        if self.resnet_model == "RN18":
            base = models.resnet18(pretrained=self.pretrained)
        d_in = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        if self.freeze_bb:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(nn.Module):
    def __init__(self, pretrained: str, d_in, d_out: int, freeze_bb: bool) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.freeze_bb = freeze_bb
        self.base = AutoModel.from_pretrained(self.pretrained)
        self.projection = Projection(d_in, d_out)

        if self.freeze_bb:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class Clip_de(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 resnet_model: str,
                 IN_pretrained: bool,
                 resnet_freeze_bb: bool,
                 # Text
                 text_model: str,
                 transformer_dim: int,
                 freeze_transformer: bool):
        super().__init__()

        self.vision_encoder = VisionEncoder(resnet_model, IN_pretrained, embed_dim, resnet_freeze_bb)
        self.caption_encoder = TextEncoder(text_model, transformer_dim, embed_dim, freeze_transformer)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        if image is None:
            return self.caption_encoder(text)
        elif text is None:
            return self.vision_encoder(image)
        image_features = self.vision_encoder(image)
        text_features = self.caption_encoder(text)

        return image_features, text_features, self.logit_scale.exp()