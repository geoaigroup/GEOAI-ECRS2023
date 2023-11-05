from torch import nn
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F
from .TSViT import PreNorm,FeedForward

class PromptAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,add_prompt=False,prompt_dim=1,external=False):
        super().__init__()
        self.external=external
        self.add_prompt=add_prompt
        if add_prompt:
        
            if external:
                self.prompt=nn.Parameter(torch.randn(1, prompt_dim ,dim))
            else:
               
         
                self.prompt_v=nn.Parameter(torch.randn(1, heads,prompt_dim ,dim_head ))
                self.prompt_k=nn.Parameter(torch.randn(1, heads,prompt_dim ,dim_head ))  

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def set_prompt(self):
        if self.add_prompt:
            if self.external:
                self.prompt.requires_grad_(True)
            else:
                self.prompt_v.requires_grad_(True)
                self.prompt_k.requires_grad_(True)
    def forward(self, x):
        if self.add_prompt and self.external: 
          prompt_expanded = repeat(self.prompt, '() N d -> b N d', b=x.shape[0])
          x = torch.cat((x,prompt_expanded), dim=1) 

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.add_prompt and not self.external:
          prompt_v_expanded = repeat(self.prompt_v, '() h N d -> b h N d', b=b)
          prompt_k_expanded = repeat(self.prompt_k, '() h N d -> b h N d', b=b)
          
          
          v = torch.cat((v,prompt_v_expanded), dim=2) 
          k = torch.cat((k,prompt_k_expanded), dim=2) 

        # print(q.shape, k.shape, v.shape)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # print(dots.shape)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class PromptTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,deep_prompt=False,prompt_dim=1,is_prompt=True,external=True):
        super().__init__()
        # self.external=external

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PromptAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,add_prompt=(is_prompt and (deep_prompt or i==0)),prompt_dim=prompt_dim,external=external)), #modified
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def set_prompt(self):
      for attn,ff in self.layers:
        attn.fn.set_prompt()

    def forward(self, x):
        dim=x.shape[1]
        for attn, ff in self.layers:
            x = attn(x)[:,:dim] + x 
            x = ff(x) + x
        return self.norm(x)
    
class PTTSViT(nn.Module):

    def __init__(self, model_config,deep_prompt=False,temp_prompt_dim=4,spa_prompt_dim=0,external=True):
        super().__init__()
        print(external)
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
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = PromptTransformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout,deep_prompt,temp_prompt_dim,temp_prompt_dim!=0,external)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = PromptTransformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout,deep_prompt,spa_prompt_dim,spa_prompt_dim!=0,external)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )

        self.mlp_head.requires_grad_(True)
        self.mlp_change=nn.Linear(in_features=self.num_classes,out_features=1)

    def set_pt_paramters(self):
      
      self.temporal_transformer.set_prompt()
      self.space_transformer.set_prompt()
      self.mlp_head.requires_grad_(True)
      self.mlp_change.requires_grad_(True)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temp1=self.to_temporal_embedding_input(xt)
        temporal_pos_embedding = temp1.reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding 
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x=self.mlp_change(x)
        x = x.permute(0, 3, 1, 2)
        return x