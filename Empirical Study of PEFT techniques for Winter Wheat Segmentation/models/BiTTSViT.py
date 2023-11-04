import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum


class BitPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def set_bias_grad(self,grad=True,gradNonPar=None):
        self.norm.bias.requires_grad = grad
        self.fn.set_bias_grad(grad,gradNonPar)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class BitFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def set_bias_grad(self,grad=True,gradNonPar=None):
        if gradNonPar is None:
            gradNonPar=grad
        self.net[0].bias.requires_grad = grad
        self.net[3].bias.requires_grad = gradNonPar
        

    def forward(self, x):
        return self.net(x)


class BitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv=nn.Linear(dim, inner_dim * 3)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def set_bias_grad(self,grad=True,gradNonPar=None):
        if gradNonPar is None:
            gradNonPar=grad
        self.to_out[0].bias.requires_grad = grad
        self.to_qkv.bias.requires_grad = gradNonPar

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class BitTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BitPreNorm(dim, BitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                BitPreNorm(dim, BitFeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def set_bias_grad(self,grad=True,gradNonPar=None):
        for attn, ff in self.layers:
            attn.set_bias_grad(grad,gradNonPar)
            ff.set_bias_grad(grad,gradNonPar)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BitFTSViT(nn.Module):
    """
    Copy of TSViT with a method to set grad for all bias and the mlp_change_head
    """

    def __init__(self, model_config,number_of_classes=1):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim), )
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = BitTransformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                   self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = BitTransformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )
        self.mlp_change = nn.Linear(in_features=self.num_classes, out_features=number_of_classes)  # added

    def set_bias_grad(self,grad=True,Full=True):
        self.to_patch_embedding[1].bias.requires_grad = grad
        self.mlp_change.requires_grad_(grad or Full)
        self.to_temporal_embedding_input.bias.requires_grad=Full
        self.mlp_head[1].bias.requires_grad=Full
        self.temporal_transformer.set_bias_grad(grad,Full)
        self.space_transformer.set_bias_grad(grad,Full)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temp1 = self.to_temporal_embedding_input(xt)
        temporal_pos_embedding = temp1.reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)  # b=B*h*w
        x = torch.cat((cls_temporal_tokens, x), dim=1)  
        x = self.temporal_transformer(x)  
        x = x[:, :self.num_classes]  
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        x += self.space_pos_embedding  
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = self.mlp_change(x)
        x = x.permute(0, 3, 1, 2)
        return x


