from torch import nn
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F
from .TSViT import PreNorm,FeedForward,Attention

class AdapterLayer(nn.Module):
    def __init__(self, dim,adapter_dim=1):
        super().__init__()
        
        self.mlp_down=nn.Linear(dim,adapter_dim)
        self.mlp_up=nn.Linear(adapter_dim,dim)
          

    def forward(self, x):
        x=self.mlp_down(x)
        x=F.relu(x)
        x=self.mlp_up(x)
        return x
    
class AdaptTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,add_adapter=False,adapter_dim=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.add_adapter=add_adapter
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                AdapterLayer(dim,adapter_dim=adapter_dim) if add_adapter else None
            ]))

    def set_params(self,required=True):
      if self.add_adapter:
        for attn,ff,adpt in self.layers:
            adpt.requires_grad_(required)


    def forward(self, x):
        for attn, ff,adpt in self.layers:
            x = attn(x) + x 
            if self.add_adapter:
                x=x+ff(x) + adpt(x)
            else:
                x = ff(x) + x
        return self.norm(x)
    
class AdaptTSViT(nn.Module):

    def __init__(self, model_config,
                 temp_adapter_dim=4,
                 spa_adapter_dim=0,
                 number_of_classes=1):
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
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = AdaptTransformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout,temp_adapter_dim!=0,temp_adapter_dim)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = AdaptTransformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout,spa_adapter_dim!=0,spa_adapter_dim)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )

        self.mlp_head.requires_grad_(True)
        self.mlp_change=nn.Linear(in_features=self.num_classes,out_features=number_of_classes)

    def set_pt_paramters(self,train=True):
      """set"""
      self.temporal_transformer.set_params(train)
      self.space_transformer.set_params(train)
      self.mlp_head.requires_grad_(train)
      self.mlp_change.requires_grad_(train)

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