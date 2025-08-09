# models_vit.py
from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer as tvit
from timm.models.vision_transformer import Attention, Mlp, DropPath

# ========== Adapter 模块（稳定版） ==========
class Adapter(nn.Module):
    """
    稳定版 Adapter:
    - 在内部先做 LayerNorm 以稳定输入分布
    - 下采样 -> 激活 -> 上采样
    - scale 参数初始化为 0，使得初始时 Adapter 等同于恒等（不会破坏预训练特征）
    """
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        hidden_dim = max(1, dim // reduction)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        # 初始化为 0：训练开始时 Adapter 不改变特征，稳定微调
        self.scale = nn.Parameter(torch.zeros(1))

        # 可选：良好初始化 adapter 的最后一层权重为 0（进一步保证初始影响为 0）
        try:
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        except Exception:
            pass

    def forward(self, x):
        # x: (B, N, C)
        y = self.ln(x)
        y = self.adapter(y)
        return x + self.scale * y


# ========== Transformer Block with Adapter ==========
class BlockWithAdapter(nn.Module):
    """
    改进点：
    - 保持原有 Attention & MLP 流程
    - 仅在 MLP 输出后插入一个 Adapter（而不是两处）
    - Adapter 接受 LayerNorm(x) 内部处理，残差相加（scale init=0）
    """
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

        # 仅保留一个 Adapter（在 MLP 后）
        self.adapter = Adapter(dim, reduction=reduction)

    def forward(self, x):
        # 注意：保持原始 block 的残差结构
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # Adapter 使用内部 LayerNorm 并以 scale 残差加入
        x = self.adapter(x)
        return x


# ========== VisionTransformer 子类（替换 blocks） ==========
class VisionTransformer(tvit.VisionTransformer):
    """
    继承 timm 的 VisionTransformer，并将 blocks 替换为 BlockWithAdapter。
    参数与 timm 保持兼容；增加 reduction 参数（Adapter 下采样比）。
    """
    def __init__(self, global_pool: bool = False, reduction: int = 16, **kwargs):
        # kwargs 应包括 embed_dim, depth, num_heads 等 timm 所需参数
        super().__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            embed_dim = kwargs['embed_dim']
            norm_layer = kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # 读取必要超参（从 kwargs 或已初始化的属性）
        depth = kwargs.get('depth', getattr(self, 'depth', 12))
        embed_dim = kwargs.get('embed_dim', getattr(self, 'embed_dim', 768))
        num_heads = kwargs.get('num_heads', getattr(self, 'num_heads', 12))
        mlp_ratio = kwargs.get('mlp_ratio', getattr(self, 'mlp_ratio', 4.0))
        qkv_bias = kwargs.get('qkv_bias', True)
        drop_rate = kwargs.get('drop_rate', getattr(self, 'drop_rate', 0.0))
        attn_drop_rate = kwargs.get('attn_drop_rate', getattr(self, 'attn_drop_rate', 0.0))
        drop_path_rate = kwargs.get('drop_path_rate', getattr(self, 'drop_path_rate', 0.0))

        # 用自定义 BlockWithAdapter 替换
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

        # 重新初始化参数（沿用 timm 的初始化方法）
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 遍历 blocks（Sequential）
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


# ========== 构造函数（保留兼容性） ==========
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


# ========== 训练/微调策略辅助函数 ==========
def freeze_backbone(model: nn.Module):
    """
    冻结 backbone（除了 head/classifier）
    假设模型有属性 .head 或 .head.fc / .head_dist 等，请根据实际命名调整。
    """
    for name, param in model.named_parameters():
        # 不冻结分类头（常见命名：head, head_dist, classifier 等）
        if 'head' in name or 'classifier' in name or 'fc_norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_adapters(model: nn.Module):
    """
    只解冻 Adapter 模块和分类头（按模块名识别 'adapter'）
    """
    for name, param in model.named_parameters():
        if 'adapter' in name or 'head' in name or 'classifier' in name or 'fc_norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def progressive_unfreeze(model: nn.Module, unfreeze_layers: int = 2):
    """
    逐层解冻：从最顶层开始解冻指定数量的 transformer block。
    unfreeze_layers: 要解冻的 block 数（从靠近输出端开始）
    说明：blocks 存在 model.blocks (nn.Sequential)
    """
    # 首先冻结所有
    for p in model.parameters():
        p.requires_grad = False

    # 解冻 head
    for name, param in model.named_parameters():
        if 'head' in name or 'classifier' in name or 'fc_norm' in name:
            param.requires_grad = True

    # blocks 是 nn.Sequential，按索引解冻后面的几层
    if hasattr(model, 'blocks'):
        total = len(model.blocks)
        # 解冻最靠近输出的 unfreeze_layers 层
        start = max(0, total - unfreeze_layers)
        for i in range(start, total):
            blk = model.blocks[i]
            for p in blk.parameters():
                p.requires_grad = True

        # 同时解冻 adapters（如果存在）
        for name, param in model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True