import torch
import loralib as lora
import pickle

from .TSViT import *
from .FTSViT import FTTSViT
from .Token_TSViT import Token_TSViT
from .PromptTuning import PTTSViT
from .adaptformer import AdaptTSViT
from .BiTTSViT import BitFTSViT
from .Adamix import AdamixTSViT
from .loraTSViT import LoraTSViT

import sys
sys.path.append("..")

from datasets_class.munich_dataset import munich_dataset



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

class MODEL_TYPE:
    """
    This class holds all model types as well as their string format
    It is like a enum in other languages
    """
    INITIAL_TSVIT=0
    RANDOM_FTSVIT=1
    HEAD_FTSVIT=2
    FULL_FTSVIT=3
    DEEP_PTTSViT=4
    SHALLOW_PTTSVIT=5
    ADAPTSViT=6
    LORA=7
    TOKEN_FTTSVIT=8
    TOKEN_EXTEND_FTTSVIT=9
    FULL_BIT_TUNE=10
    PARTIAL_BIT_TUNE=11
    ADAMIXTSVIT=12
    def to_string(self,model):
        names={
            self.INITIAL_TSVIT:"INITIAL_TSVIT",
            self.RANDOM_FTSVIT:"RANDOM_FTSVIT",
            self.HEAD_FTSVIT:"HEAD_FTSVIT",
            self.FULL_FTSVIT:"FULL_FTSVIT",
            self.DEEP_PTTSViT:"DEEP_PTTSViT",
            self.SHALLOW_PTTSVIT:"SHALLOW_PTTSVIT",
            self.ADAPTSViT:"ADAPTSViT",
            self.LORA:"LORA",
            self.TOKEN_FTTSVIT:"TOKEN_FTTSVIT",
            self.TOKEN_EXTEND_FTTSVIT:"TOKEN_EXTEND_FTTSVIT",
            self.FULL_BIT_TUNE:"FULL_BIT_TUNE", 
            self.PARTIAL_BIT_TUNE:"PARTIAL_BIT_TUNE", 
            self.ADAMIXTSVIT:"ADAMIXTSVIT", 

        }
        return names[model]

def create_model(configuration):
    """
    This function create a model for training by using the configurations given
    the configurations
    """
    tempTSViTConfig=TSVIT_config.copy()
    net=TSViT(tempTSViTConfig)
    net.load_state_dict(torch.load(configuration["initial_weight_file"]))
    model_number=configuration["model_number"]
    if model_number==MODEL_TYPE.RANDOM_FTSVIT:
        net=FTTSViT(TSViT(tempTSViTConfig),number_of_classes=configuration["number_of_classes"])
        net.requires_grad_(True)
    elif model_number==MODEL_TYPE.FULL_FTSVIT:
        net=FTTSViT(net,number_of_classes=configuration["number_of_classes"])
        net.requires_grad_(True)
    
    elif model_number==MODEL_TYPE.HEAD_FTSVIT:
        net.requires_grad_(False)
        net=FTTSViT(net,number_of_classes=configuration["number_of_classes"])
    
    elif model_number==MODEL_TYPE.SHALLOW_PTTSVIT or model_number==MODEL_TYPE.DEEP_PTTSViT:
        if "external" not in configuration.keys():
            configuration["external"]=True
        if configuration["change_to_token"]:
            tempTSViTConfig["num_classes"]=configuration["number_of_classes"]
        
        net1=PTTSViT(tempTSViTConfig,model_number==MODEL_TYPE.DEEP_PTTSViT,configuration["temporal_prompt_dim"],configuration["spatial_prompt_dim"],configuration["external"])
        
        if configuration["change_to_token"]:
            net.temporal_token=None
        net1.load_state_dict(net.state_dict(),strict=False)
        net=net1
        net.requires_grad_(False)
        net.set_pt_paramters()
        if configuration["change_to_token"]:
            net.mlp_change=nn.Identity()
    elif model_number==MODEL_TYPE.ADAPTSViT:
        if configuration["change_to_token"]:
            tempTSViTConfig["num_classes"]=configuration["number_of_classes"]
        net1=AdaptTSViT(tempTSViTConfig,configuration["temporal_adapter_dim"],configuration["spatial_adapter_dim"],number_of_classes=configuration["number_of_classes"])
        if configuration["change_to_token"]:
            net.temporal_token=None
        net1.load_state_dict(net.state_dict(),strict=False)
        net=net1
        net.requires_grad_(False)
        net.set_pt_paramters()
        net.temporal_token.requires_grad_(True)
        if configuration["change_to_token"]:
            net.mlp_change=nn.Identity()
    
    elif model_number==MODEL_TYPE.LORA:
        if "rt" not in configuration.keys():
            configuration["rt"]=None
        if "rs" not in configuration.keys():
            configuration["rs"]=None
        if configuration["change_to_token"]:
            tempTSViTConfig["num_classes"]=configuration["number_of_classes"]
            net.temporal_token=None
            
        net1=LoraTSViT(tempTSViTConfig,r=configuration["r"],rt=configuration["rt"],rs=configuration["rs"],number_of_classes=configuration["number_of_classes"])
        net1.load_state_dict(net.state_dict(),strict=False)
        lora.mark_only_lora_as_trainable(net1)
        net=net1
        net.mlp_change.requires_grad_(True)
        if configuration["change_to_token"]:
            net.mlp_change=nn.Identity()
            net.temporal_token=nn.Parameter(torch.randn(1, configuration["number_of_classes"], 128))
    elif model_number==MODEL_TYPE.INITIAL_TSVIT:
         net=TSViT(configuration["model_config"])
    
    elif model_number==MODEL_TYPE.TOKEN_FTTSVIT:
        net1=TSViT(configuration["model_config"])
        net.temporal_token=None
        net=net1
        if "full" not in configuration.keys():
            configuration["full"]=False
        net.requires_grad_(configuration["full"])
        net.temporal_token.requires_grad_(True)
    elif model_number==MODEL_TYPE.TOKEN_EXTEND_FTTSVIT:
        net1=Token_TSViT(configuration["model_config"])
        net1.load_state_dict(net.state_dict(),strict=False)
        net=net1
        net.requires_grad_(False)
        net.temporal_token.requires_grad_(configuration["all_tokens"])
        net.temporal_token1.requires_grad_(True)
    elif model_number==MODEL_TYPE.FULL_BIT_TUNE or model_number==MODEL_TYPE.PARTIAL_BIT_TUNE:

        if configuration["change_to_token"]:
            tempTSViTConfig["num_classes"]=configuration["number_of_classes"]
            net.temporal_token=None

        net1=BitFTSViT(model_config=tempTSViTConfig,number_of_classes=configuration["number_of_classes"])
        net1.requires_grad_(False)
        net1.set_bias_grad(True,model_number==MODEL_TYPE.FULL_BIT_TUNE)
        net1.load_state_dict(net.state_dict(),strict=False)
        net=net1

        if configuration["change_to_token"]:
            net.mlp_change=nn.Identity()
            net.temporal_token=nn.Parameter(torch.randn(1, configuration["number_of_classes"], 128))

    elif model_number==MODEL_TYPE.ADAMIXTSVIT:
        net1=AdamixTSViT(tempTSViTConfig,model_number==4,configuration["temporal_adapter_dim"],configuration["spatial_adapter_dim"])
        net1.load_state_dict(net.state_dict(),strict=False)
        net=net1
        net.requires_grad_(False)
        net.set_pt_paramters()     
    return net

def load_model_from_name(name,option="best_f1score.pt"):
    config=pickle.load(open(f"save_models/{name}/config.pkl","rb"))
    net=create_model(configuration=config)
    net.load_state_dict(torch.load(f"save_models/{name}/{option}"))
    return net

# def load_model_from(configurations=None,path=None):
#     pass