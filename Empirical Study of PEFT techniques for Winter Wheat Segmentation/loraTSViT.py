from torch import nn
import loralib as lora
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F

class LoraPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class LoraFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.,r=-1):
        assert r!=-1

        super().__init__()
        self.net = nn.Sequential(
            lora.Linear(dim, hidden_dim,r=r) if r!=0 else nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            lora.Linear(hidden_dim, dim,r=r) if r!=0 else nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class LoraAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,r=-1):
        assert r>=0
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = lora.Linear(dim, inner_dim * 3, bias=False,r=r)  if r!=0 else nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            lora.Linear(inner_dim, dim,r=r) if r!=0 else nn.Linear(dim, inner_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class LoraTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,r=-1):
        assert r>=0
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LoraPreNorm(dim, LoraAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,r=r)),
                LoraPreNorm(dim, LoraFeedForward(dim, mlp_dim, dropout=dropout,r=r))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class LoraTSViT(nn.Module):
    def __init__(self, model_config,r=1,rt=None,rs=None,number_of_classes=1):
        
        if rt is None:
            rt=r
        if rs is None:
            rs=r

        assert rt>-1
        assert rs>-1
        
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
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
            lora.Linear(patch_dim, self.dim,r=r) if r!=0 else nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = lora.Linear(366, self.dim,r=r) if r!=0 else nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_LoraTransformer = LoraTransformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout,r=rt)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_LoraTransformer = LoraTransformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout,r=rs)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            lora.Linear(self.dim, self.patch_size**2,r=r) if r!=0 else nn.Linear(self.dim, self.patch_size**2)
        )
        self.mlp_head.requires_grad_(True)#added
        self.mlp_change=nn.Linear(in_features=self.num_classes,out_features=number_of_classes)#added

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        
        # print(xt.shape)
        
        xt = xt.reshape(-1, 366)
        # print(xt.size())
        temp1=self.to_temporal_embedding_input(xt)
        # print(temp1.size())
        temporal_pos_embedding = temp1.reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        # print(x.size())
        x += temporal_pos_embedding.unsqueeze(1)
        # print(x.size())
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)#b=B*h*w
        # print(x.size())
        x = torch.cat((cls_temporal_tokens, x), dim=1) #(b*h*w, token+T,dim)
        # print(x.size())
        x = self.temporal_LoraTransformer(x)#(b*h*w, token+T,dim)
        # print(x.size())

        
        x = x[:, :self.num_classes]#(b*h*w, token ,dim)
        
        
        # print(x.size())
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_LoraTransformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        
        x=self.mlp_change(x)
        
        x = x.permute(0, 3, 1, 2)
        return x


if __name__=="__main__":
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    TSVIT_config={
        'img_res':24,
        'patch_size':2,
        'num_classes':19,
        "max_seq_len":60,
        'dim':128,
        'temporal_depth':4,
        'spatial_depth': 4,
        'heads': 4,
        'pool': 'cls',
        'dim_head': 32,
        'emb_dropout': 0.,
        'scale_dim': 4,
        'dropout':0,
        'num_channels': 11,
        'num_feature':16,
        'scale_dim':4,
        'ignore_background': False
    }
    net=LoraTSViT(TSVIT_config,0,2,0)
    # net.requires_grad_(False)
    lora.mark_only_lora_as_trainable(net)
    net.mlp_change.requires_grad_(True)
    print_trainable_parameters(net)
    # print(lora)c

