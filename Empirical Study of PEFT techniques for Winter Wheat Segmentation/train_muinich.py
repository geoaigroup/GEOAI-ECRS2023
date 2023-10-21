import pickle
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from TSViT import TSViT
from FTSViT import FTTSViT
from Token_TSViT import Token_TSViT
from PromptTuning import PTTSViT
from util import print_trainable_parameters,get_trainable_parameters,loss_and_metrics_multi_class,F1score
import numpy as np
from tqdm import tqdm
from TSViT import TSViT,Transformer,PreNorm,Attention,FeedForward
import os
import neptune
from adaptformer import AdaptTSViT
from loraTSViT import LoraTSViT
import loralib as lora
import random

from loss import DiceBCELoss

TSVIT_config={
        'img_res':24,
        'patch_size':2,
        'num_classes':27,#19,
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

# model_type=["from_scratch","fine tune full","head_fine_tune","shallow_prompt_tune v2","deep_prompt_tune v2","adapter tune","lora tune"]

class model_type:
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
        }
        return names[model]


def load_model(configuration):
        net=torch.load(configuration["initial_wait_file"])
        model_number=configuration["model_number"]
        if model_number==model_type.RANDOM_FTSVIT:
            net=FTTSViT(TSViT(TSVIT_config),number_of_classes=27)
            net.requires_grad_(True)

        elif model_number==model_type.FULL_FTSVIT:
            net=FTTSViT(net,number_of_classes=27)
            net.requires_grad_(True)
        
        elif model_number==model_type.HEAD_FTSVIT:
            net.requires_grad_(False)
            net=FTTSViT(net,number_of_classes=27)
        
        elif model_number==model_type.SHALLOW_PTTSVIT or model_number==model_type.DEEP_PTTSViT:
            if "external" not in configuration.keys():
                configuration["external"]=True

            if not configuration["change_to_token"]:
                TSVIT_config["num_classes"]=19
            
            net1=PTTSViT(TSVIT_config,model_number==model_type.DEEP_PTTSViT,configuration["temporal_prompt_dim"],configuration["spatial_prompt_dim"],configuration["external"])
            
            if configuration["change_to_token"]:
                net.temporal_token=None

            net1.load_state_dict(net.state_dict(),strict=False)
            net=net1
            net.requires_grad_(False)
            net.set_pt_paramters()
            if configuration["change_to_token"]:
                net.mlp_change=nn.Identity()

        elif model_number==model_type.ADAPTSViT:
            if not configuration["change_to_token"]:
                 TSVIT_config["num_classes"]=19
            net1=AdaptTSViT(TSVIT_config,configuration["temporal_adapter_dim"],configuration["spatial_adapter_dim"],number_of_classes=27)
            if configuration["change_to_token"]:
                net.temporal_token=None
            net1.load_state_dict(net.state_dict(),strict=False)
            net=net1
            net.requires_grad_(False)
            net.set_pt_paramters()
            net.temporal_token.requires_grad_(True)
            if configuration["change_to_token"]:
                net.mlp_change=nn.Identity()
        
        elif model_number==model_type.LORA:
            if "rt" not in configuration.keys():
                configuration["rt"]=None
            if "rs" not in configuration.keys():
                configuration["rs"]=None
            if not configuration["change_to_token"]:
                 TSVIT_config["num_classes"]=19
            else:
                 net.temporal_token=None
            net1=LoraTSViT(TSVIT_config,r=configuration["r"],rt=configuration["rt"],rs=configuration["rs"],number_of_classes=27)
            net1.load_state_dict(net.state_dict(),strict=False)
            lora.mark_only_lora_as_trainable(net1)
            net=net1
            net.mlp_change.requires_grad_(True)
            if configuration["change_to_token"]:
                net.mlp_change=nn.Identity()
                net.temporal_token=nn.Parameter(torch.randn(1, 27, 128))
        elif model_number==model_type.INITIAL_TSVIT:
             net=TSViT(configuration["model_config"])
        
        elif model_number==model_type.TOKEN_FTTSVIT:
            net1=TSViT(configuration["model_config"])
            net.temporal_token=None
            print(net1.load_state_dict(net.state_dict(),strict=False))
            net=net1
            if "full" not in configuration.keys():
                configuration["full"]=False
            net.requires_grad_(configuration["full"])
            net.temporal_token.requires_grad_(True)

        elif model_number==model_type.TOKEN_EXTEND_FTTSVIT:
            net1=Token_TSViT(configuration["model_config"])
            print(net1.load_state_dict(net.state_dict(),strict=False))
            net=net1
            net.requires_grad_(False)
            net.temporal_token.requires_grad_(configuration["all_tokens"])
            net.temporal_token1.requires_grad_(True)

             
        
        return net


def test(net,dataset):
        
        temp_eval_loss=[]
        with torch.no_grad():
            net.eval()
            for X,y in tqdm(DataLoader(dataset,batch_size=16,num_workers=2)):
                with torch.no_grad():
                    X,y=X.cuda(),y.cuda()
                    yp=net(X)
                    temp_eval_loss.append(loss_and_metrics_multi_class(y.cpu().detach().to(torch.float).numpy(),yp.to(torch.float).cpu().detach().numpy(),criterion=torch.nn.CrossEntropyLoss()))

            test_loss=np.array(temp_eval_loss).mean(axis=0)
            print("test loss= ",test_loss)
            return test_loss
         
def set_seed(seed=911):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def PEFTTrain ( peft_config):

    set_seed(peft_config["seed"])

    train_loss=[]
    eval_loss=[]


    train_dataset=peft_config["train_dataset"] 
    eval_dataset=peft_config["eval_dataset"] 
    test_dataset=peft_config["test_dataset"] 
    
    print("loading the model:")
    net=load_model(peft_config)
        

    best_f1score=0
    best_iou=0
    best_loss=10
    
    model_name=peft_config["model_name"]

    net.cuda()
    print_trainable_parameters(TSViT(TSVIT_config)),print_trainable_parameters(net)
    net=net.train()

    optimizer=torch.optim.Adam(get_trainable_parameters(net),lr=peft_config["lr"]) #.parameters()

    if peft_config["add_scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3162,verbose=True)
    
    lossfn=nn.CrossEntropyLoss() if not ("add_dice_loss" in peft_config.keys() and peft_config["add_dice_loss"]) else DiceBCELoss() #torch.nn.CrossEntropyLoss() 
    os.makedirs(f"save_models/{model_name}")#,exist_ok=True
    pickle.dump(peft_config,open(f"save_models/{model_name}/config.pkl","wb"))
    if run_config["do_neptune"]:
        run=neptune.init_run(project="GEOgroup/crop-monitoring",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZGU2MDg4MC0yOTE5LTRjMmItYjZmMi1jNDJjMGRhYjcyZWQifQ==",
        name=model_name,
        custom_run_id=model_name)
        hyperparams={
                "learning rate":peft_config["lr"],
                "scheduler drop":"radical 10 per 2 epochs" if peft_config["add_scheduler"] else "constant",
                "model name":model_name,
                "technique":model_type.to_string(model_type,peft_config["model_number"]),
                "trainable parameters": print_trainable_parameters(net)[0],
                "total number of parameters":print_trainable_parameters(net)[1],
                "percentange of trainable parameters": print_trainable_parameters(net)[2],
                "additional parameters":print_trainable_parameters(net)[1]-print_trainable_parameters(TSViT(TSVIT_config))[1],
        }
        run["hyperparameters"] = hyperparams
        run["configuration"] = peft_config
    net.train()
    print("starting training")
    for epoch in range(peft_config["number_of_epochs"]):
        train_loss=[]
        loader=tqdm(DataLoader(train_dataset,batch_size=16, num_workers=2,shuffle=True))
        for X,y in loader:
            
            X,y=X.cuda(),y.cuda()
            optimizer.zero_grad()
            yp=net(X)


            loss=lossfn(yp,y.to(torch.long).cuda())          
            loss.backward()
            optimizer.step()
            train_loss.append(loss_and_metrics_multi_class(y.cpu().detach().to(torch.float).numpy(),yp.to(torch.float).cpu().detach().numpy()))

            loader.set_description(str(train_loss[-1]))
        train_loss=np.array(train_loss).mean(axis=0)
        print(train_loss)
        if run_config["do_neptune"]:        
                run["train/batch/loss"].log(train_loss[0])
                run["train/batch/acc"].log(train_loss[1])
                run["train/batch/iou"].log(train_loss[2])
                run["train/batch/f1score"].log(train_loss[3])
                
        temp_eval_loss=[]

        if epoch==peft_config["number_of_epochs"]-1:
                torch.save(net.state_dict(),f"save_models/{model_name}/epoch_{epoch}.pt")
        with torch.no_grad():
            net.eval()
            for X,y in tqdm(DataLoader(eval_dataset,batch_size=16,num_workers=2)):
                with torch.no_grad():
                    X,y=X.cuda(),y.cuda()
                    yp=net(X)
                    loss=lossfn(yp,y.to(torch.long).cuda())
                    temp_eval_loss.append(loss_and_metrics_multi_class(y.cpu().detach().to(torch.float).numpy(),yp.to(torch.float).cpu().detach().numpy()))
                    # print(np.array(temp_eval_loss).mean())
            
            eval_loss.append(np.array(temp_eval_loss).mean(axis=0))
            print("eval",epoch,eval_loss[-1])
            if run_config["do_neptune"]:        
                run["eval/batch/loss"].log(eval_loss[-1][0])
                run["eval/batch/acc"].log(eval_loss[-1][1])
                run["eval/batch/iou"].log(eval_loss[-1][2])
                run["eval/batch/f1score"].log(eval_loss[-1][3])

            print("eval loss= ",eval_loss[-1])
            print("end of epoch",epoch)
            if(best_loss>eval_loss[-1][0]):
                        best_loss=eval_loss[-1][0]
                        torch.save(net.state_dict(),f"save_models/{model_name}/best_loss.pt")

            if(best_iou<eval_loss[-1][2]):
                        best_loss=eval_loss[-1][2]
                        torch.save(net.state_dict(),f"save_models/{model_name}/best_iou.pt")

            if(best_f1score<eval_loss[-1][3]):
                        best_loss=eval_loss[-1][3]
                        torch.save(net.state_dict(),f"save_models/{model_name}/best_f1score.pt")
                        
            # if run_config["do_neptune"]:
            #     run["validation/epoch/loss"].log(eval_loss[-1])
            net.train()
            if peft_config["add_scheduler"]:
                        scheduler.step()
                        
    net.eval()


    print("starting testing")
    print("testing final epoch")
    test_loss_last =test(net,test_dataset)
    if run_config["do_neptune"]:
        run["test/last_epoch/loss"]=(test_loss_last[0])
        run["test/last_epoch/acc"]=(test_loss_last[1])
        run["test/last_epoch/iou"]=(test_loss_last[2])
        run["test/last_epoch/f1score"]=(test_loss_last[3])


    print("testing best eval f1 score")
    net.load_state_dict(torch.load(f"save_models/{model_name}/best_f1score.pt"))
    
    test_loss_best_f1 =test(net,test_dataset) 
    if run_config["do_neptune"]:
        run["test/best_f1/loss"]=(test_loss_best_f1[0])
        run["test/best_f1/acc"]=(test_loss_best_f1[1])
        run["test/best_f1/iou"]=(test_loss_best_f1[2])
        run["test/best_f1/f1score"]=(test_loss_best_f1[3])

    print("testing best eval loss")
    net.load_state_dict(torch.load(f"save_models/{model_name}/best_loss.pt"))
    test_loss_best_f1 =test(net,test_dataset)
    if run_config["do_neptune"]:
        run["test/best_loss/loss"]=(test_loss_best_f1[0])
        run["test/best_loss/acc"]=(test_loss_best_f1[1])
        run["test/best_loss/iou"]=(test_loss_best_f1[2])
        run["test/best_loss/f1score"]=(test_loss_best_f1[3])


    if run_config["do_neptune"]:
        run.stop()



if __name__=="__main__":
    my_config={
        'img_res':24,
        'patch_size':2,
        'num_classes':27,
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
    from munich_dataset import munich_dataset
    base_path="D:\\GEOAI\\datasets\\munich_24"
    

    # import random 
    def get_tile(name):
        return name.split("_")[1]
    
    # train_datalist=[a for a in  os.listdir(base_path) if a.startswith("16_")]

    # test_datalist=[a for a in  os.listdir(base_path) if a.startswith("17_")]

    # train_dataset=munich_dataset(base_path,train_datalist)
    # eval_dataset=munich_dataset(base_path,test_datalist[:3200])
    # test_dataset=munich_dataset(base_path,test_datalist[3200:])


    datalist=[a for a in  os.listdir(base_path)]
    train_path="D:\\GEOAI\\datasets\\train_muinich\\original_dist\\train_fold0.tileids"
    with open(train_path,"r") as f:
        train_id=f.read().split("\n")
    train_datalist=[a for a in datalist if get_tile(a) in train_id]
    test_path="D:\\GEOAI\\datasets\\train_muinich\\original_dist\\test_fold0.tileids"
    with open(test_path,"r") as f:
        test_id=f.read().split("\n")
    test_datalist=[a for a in datalist if get_tile(a) in test_id]
    eval_path="D:\\GEOAI\\datasets\\train_muinich\\original_dist\\eval.tileids"
    with open(eval_path,"r") as f:
        eval_id=f.read().split("\n")
    eval_datalist=[a for a in datalist if get_tile(a) in eval_id]

    train_dataset=munich_dataset(base_path,train_datalist)
    eval_dataset=munich_dataset(base_path,eval_datalist)
    test_dataset=munich_dataset(base_path,test_datalist)

    run_config={
        "train_dataset":train_dataset,
        "test_dataset":test_dataset,
        "eval_dataset":eval_dataset,
         
        "model_number":model_type.LORA,


        "change_to_token":True,


        "model_name":"tesyibgss33",
        "initial_wait_file":"D:\\GEOAI\\datasets\\train_muinich\\model.pt",

        "lr":1e-3,
        "number_of_epochs":40,

        # "full":True,

        "r":1,
        "rs":1,
        "rt":2,
        
        "external":True,
        "temporal_prompt_dim":2,
        "spatial_prompt_dim":2,
        # "temporal_adapter_dim":8,
        # "spatial_adapter_dim":8,

        "all_tokens":True,
        "model_config":my_config,

        "test_and_eval_split":True,
        "no_eval":False,
        "do_neptune":True,#False,
        "add_scheduler":False,
        "vertical_flip":False,
        "horizontal_flip":False,
        "semisupervised":False,
        "seed":313,
    }
    PEFTTrain(run_config)

  