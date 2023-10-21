from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange,repeat

from TSViT import TSViT 

class FTTSViT(nn.Module):

    def __init__(self, tsvit:TSViT,tsvit_require_grad=False,number_of_classes=1):
        super().__init__()
        tsvit.requires_grad_(tsvit_require_grad)
        self.image_size =tsvit.image_size
        self.patch_size =tsvit.patch_size
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes =tsvit.num_classes
        self.num_frames =tsvit.num_frames
        self.dim =tsvit.dim
        self.temporal_depth=tsvit.temporal_depth
        self.spatial_depth=tsvit.spatial_depth
        self.heads =tsvit.heads
        self.dim_head =tsvit.dim_head
        self.dropout =tsvit.dropout
        self.emb_dropout =tsvit.emb_dropout
        self.pool = tsvit.pool
        self.scale_dim =tsvit.scale_dim
        num_patches = self.num_patches_1d ** 2
        self.to_patch_embedding =tsvit.to_patch_embedding
        self.to_temporal_embedding_input = tsvit.to_temporal_embedding_input
        
        
        
        self.temporal_token =tsvit.temporal_token

        self.temporal_transformer =tsvit.temporal_transformer 

        self.space_pos_embedding =tsvit.space_pos_embedding 

        
        
        self.space_transformer =tsvit.space_transformer
        self.dropout =tsvit.dropout
        self.mlp_head =tsvit.mlp_head

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

        # temp2=self.to_temporal_embedding_input_pt(xt)#added

        # print(temp1.size())
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