from torch import nn
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F
from TSViT import PreNorm,FeedForward,Attention

class AdapterLayer(nn.Module):
    def __init__(self, dim, adapter_dim):
        super().__init__()

        self.mlp_layer = nn.Linear(dim, adapter_dim)
        self.weight=self.mlp_layer.weight
        self.bias=self.mlp_layer.bias

    def forward(self, x):
        x = self.mlp_layer(x)
        return x


class AdapterLayers(nn.Module):
    def __init__(self, dim, adapter_dim=1):
        super().__init__()

        self.down_layers = nn.ModuleList([AdapterLayer(dim, adapter_dim) for i in range(2)])
        #self.up_layers = nn.ModuleList([AdapterLayer(adapter_dim,dim) for i in range(2)])
        self.up_layer = AdapterLayer(adapter_dim,dim)
        with torch.no_grad():
             self.test_down=nn.Linear(dim, adapter_dim)
             #self.test_up=nn.Linear(adapter_dim,dim)

    def forward(self, x,test=False):
        if(test==False):
            down_idx = torch.randint(low=0, high=2, size=(1,)).item()
            #up_idx = torch.randint(low=0, high=2, size=(1,)).item()
            x = self.down_layers[down_idx].forward(x)
            x=  F.relu(x)
            #x = self.up_layers[up_idx].forward(x)
            x = self.up_layer.forward(x)
        else:
            with torch.no_grad():
                #self.test_up.weight.data.zero_()
                #self.test_up.bias.data.zero_()
                self.test_down.weight.data.zero_()
                self.test_down.bias.data.zero_()
                for i in range(2):
                     self.test_down.weight += 0.5*self.down_layers[i].weight
                     #self.test_up.weight += 0.5*self.up_layers[i].weight
                     self.test_down.bias += 0.5*self.down_layers[i].bias
                     #self.test_up.bias += 0.5*self.up_layers[i].bias
                x = self.test_down(x)
                x=  F.relu(x)
                #x = self.test_up(x)
                x = self.up_layer.forward(x)

        return x


class AdamixTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., add_adapter=True, adapter_dim=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.add_adapter = add_adapter
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # modified
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                AdapterLayers(dim, adapter_dim=adapter_dim) if add_adapter else None
            ]))

    def set_prompt(self):
        if self.add_adapter:
            for attn, ff, adpt in self.layers:
                adpt.requires_grad_(True)

    def forward(self, x,test=False):
        for attn, ff, adpt in self.layers:
            x = attn(x) + x  # [:,:dim]
            x = x + ff(x) + adpt(x,test)
        return self.norm(x)


class AdamixTSViT(nn.Module):

    def __init__(self, model_config,
                 deep_prompt=False,
                 temp_prompt_dim=4,
                 spa_prompt_dim=0):
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
        self.temporal_transformer = AdamixTransformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                      self.dim * self.scale_dim, self.dropout, temp_prompt_dim != 0,
                                                      temp_prompt_dim)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = AdamixTransformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                                   self.dim * self.scale_dim, self.dropout, spa_prompt_dim != 0,
                                                   spa_prompt_dim)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2)
        )
        self.mlp_head.requires_grad_(False)  # added
        self.mlp_change = nn.Linear(in_features=self.num_classes, out_features=1)
        self.mlp_change.requires_grad_(False)  # added

    def set_pt_paramters(self):

        self.temporal_transformer.set_prompt()
        self.space_transformer.set_prompt()
        self.mlp_head.requires_grad_(True)
        self.mlp_change.requires_grad_(True)

    def forward(self, x,test=False):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        # print(xt.shape)

        xt = xt.reshape(-1, 366)
        # print(xt.size())
        temp1 = self.to_temporal_embedding_input(xt)

        # temp2=self.to_temporal_embedding_input_pt(xt)#added

        # print(temp1.size())
        temporal_pos_embedding = temp1.reshape(B, T, self.dim)

        # temporal_pos_embedding_pt=temp2.reshape(B,T,self.dim)#added

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)  # +temporal_pos_embedding_pt.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)

        # cls_temporal_tokens_pt = repeat(self.temporal_token_pt, '() N d -> b N d', b=B * self.num_patches_1d ** 2)#added

        # cls_spatial_tokens_pt = repeat(self.temporal_token_pt, '() N d -> b N d', b=B * self.num_classes )#added

        x = torch.cat((cls_temporal_tokens, x), dim=1)  # cls_temporal_tokens_pt,
        x = self.temporal_transformer(x,test)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim)

        x += self.space_pos_embedding  # +self.space_pos_embedding_pt #[:, :, :(n + 1)]

        x = self.dropout(x)
        x = self.space_transformer(x,test)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = self.mlp_change(x)
        x = x.permute(0, 3, 1, 2)
        return x
