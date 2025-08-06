# models_vit.py

from functools import partial

import timm.models.vision_transformer
import torch

from timm.models.vision_transformer import Attention, Mlp, DropPath, nn

# Adapter 模块
class Adapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim)
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.scale * self.adapter(x)

# 带 Adapter 的 Transformer Block
class BlockWithAdapter(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, reduction=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        self.adapter1 = Adapter(dim, reduction=reduction)
        self.adapter2 = Adapter(dim, reduction=reduction)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.adapter1(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.adapter2(x)
        return x


# 继承 timm VisionTransformer，替换blocks为自定义的BlockWithAdapter
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, reduction=16, **kwargs):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            embed_dim = kwargs['embed_dim']
            norm_layer = kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # 替换 blocks
        depth = kwargs.get('depth', 24)
        embed_dim = kwargs.get('embed_dim', 1024)
        num_heads = kwargs.get('num_heads', 16)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', 0.)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)

        self.blocks = nn.Sequential(*[
            BlockWithAdapter(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                reduction=reduction
            )
            for _ in range(depth)
        ])

        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

# RETFound_mae 创建入口，保留兼容性
def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
