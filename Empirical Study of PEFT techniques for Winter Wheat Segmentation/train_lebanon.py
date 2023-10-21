import pickle
from dataset import TSViT_leb_dataset
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from TSViT import TSViT
from FTSViT import FTTSViT
from PromptTuning import PTTSViT
from util import print_trainable_parameters,get_trainable_parameters,loss_and_metrics_single_class,F1score,set_seed
import numpy as np
from tqdm import tqdm
from TSViT import TSViT,Transformer,PreNorm,Attention,FeedForward
import os
import neptune
from adaptformer import AdaptTSViT
from loraTSViT import LoraTSViT
import loralib as lora

from loss import DiceBCELoss,semiSupervisedBCELoss

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
     FROM_SCRATCH=0
     FULL_FINE_TUNE=1
     HEAD_FINE_TUNE=2
     SHALLOW_PROMPT_TUNE=3
     DEEP_PROMPT_TUNE=4
     ADAPTER_TUNE=5
     LORA_TUNE=6
     TOKEN_TUNE=7
     model_type=["from scratch","fine tune full","head fine tune","shallow prompt tune","deep prompt tune","adapter tune","lora tune","token tune"]


def load_model(configuration):
        net=torch.load(configuration["initial_wait_file"])
        model_number=configuration["model_number"]
        if model_number==MODEL_TYPE.FROM_SCRATCH:
            net=FTTSViT(TSViT(TSVIT_config))
            net.requires_grad_(True)
        elif model_number==MODEL_TYPE.FULL_FINE_TUNE:
            net=FTTSViT(net)
            net.requires_grad_(True)
        elif model_number==MODEL_TYPE.HEAD_FINE_TUNE:
            net.requires_grad_(False)
            net=FTTSViT(net)
        elif model_number==MODEL_TYPE.DEEP_PROMPT_TUNE or model_number==MODEL_TYPE.SHALLOW_PROMPT_TUNE:
            if "external" not in configuration.keys():
                 configuration["external"]=True
            net1=PTTSViT(TSVIT_config,model_number==MODEL_TYPE.DEEP_PROMPT_TUNE,configuration["temporal_prompt_dim"],configuration["spatial_prompt_dim"],configuration["external"])
            net1.load_state_dict(net.state_dict(),strict=False)
            net=net1
            net.requires_grad_(False)
            net.set_pt_paramters()
            net.mlp_change.requires_grad_(True)
        elif model_number==MODEL_TYPE.ADAPTER_TUNE:
            net1=AdaptTSViT(TSVIT_config,configuration["temporal_adapter_dim"],configuration["spatial_adapter_dim"])
            net1.load_state_dict(net.state_dict(),strict=False)
            net=net1
            net.requires_grad_(False)
            net.set_pt_paramters()
        elif model_number==MODEL_TYPE.LORA_TUNE:
            if "rt" not in configuration.keys():
                configuration["rt"]=None
            if "rs" not in configuration.keys():
                configuration["rs"]=None
            print(configuration["rs"],configuration["rt"])
            net1=LoraTSViT(TSVIT_config,r=configuration["r"],rt=configuration["rt"],rs=configuration["rs"])
            net1.load_state_dict(net.state_dict(),strict=False)
            lora.mark_only_lora_as_trainable(net1)
            net=net1
            net.mlp_change.requires_grad_(True)
        elif model_number==MODEL_TYPE.TOKEN_TUNE:
            net1=TSViT(configuration["my_TSViT_config"])
            net.temporal_token=None
            net1.load_state_dict(net.state_dict(),strict=False)
            net=net1
            net.requires_grad_(False)
            net.temporal_token.requires_grad_(True)
        return net

def load_dataset(config):
    if "datalist_path" in config.keys():
        with open(config["datalist_path"],'rb') as f:
            datalist=pickle.load(f)
        if "train_years" in config.keys():
                print(config["train_years"])
                datalist=[data for data in datalist if data[1] in config["train_years"]]
    data_path=config["data_path"]

    vertical_flip=config["vertical_flip"] if "vertical_flip" in config else False
    horizontal_flip=config["horizontal_flip"] if "horizontal_flip" in config else False

    train_min_pixels_per_image=config["train_min_pixels_per_image"] if "train_min_pixels_per_image" in config.keys() else 1
    eval_min_pixels_per_image=config["eval_min_pixels_per_image"] if "eval_min_pixels_per_image" in config.keys() else 1

    if config["test_and_eval_split"]:
        train_dataset=TSViT_leb_dataset(data_path,list_of_tile=datalist,min_pixels_per_image=train_min_pixels_per_image,vertical_flip=vertical_flip,horizontal_flip=horizontal_flip)
        test_dataset=TSViT_leb_dataset(data_path,[2020],aois=[1,2,3,4],vertical_flip=False,horizontal_flip=False)
        eval_dataset=TSViT_leb_dataset(data_path,[2020],aois=[0],vertical_flip=False,horizontal_flip=False)
        return train_dataset,eval_dataset,test_dataset
    test_dataset=TSViT_leb_dataset(data_path,[2020],aois=[0,1,2,3,4])
    
    train_datalist=[i for i in datalist if i[1] in [2017,2018,2019]]
    test_datalist=[i for i in datalist if i[1] in [2016]]
    train_dataset=TSViT_leb_dataset(data_path,list_of_tile=train_datalist,min_pixels_per_image=train_min_pixels_per_image,vertical_flip=vertical_flip,horizontal_flip=horizontal_flip)
    eval_dataset=TSViT_leb_dataset(data_path,list_of_tile=test_datalist,min_pixels_per_image=eval_min_pixels_per_image)
    return train_dataset,eval_dataset,test_dataset

def test(net,dataset,lossfn):
        temp_eval_loss=[]
        with torch.no_grad():
            net.eval()
            dice=F1score()
            for X,y,mask in tqdm(DataLoader(dataset,batch_size=64,num_workers=2)):
                with torch.no_grad():
                        X=X.permute(0,1,3,4,2).to(torch.float).cuda()
                        yp=net(X)
                        yp=yp.squeeze()
                        y=(y[:,0]!=0)
                        mask=mask[:,0]!=0
                        yp=yp[mask]
                        y=y[mask].cuda()
                        dice(yp.cpu(),y.cpu())
                        loss=lossfn(yp,y.to(torch.float).cuda())
                        temp_eval_loss.append(loss.item())

            macro_dice=dice.Macro_f1()
            print("test dice=", macro_dice)

            test_loss=np.array(temp_eval_loss).mean()
            print("test loss= ",test_loss)

            return test_loss,macro_dice
         


def PEFTTrain ( peft_config):
    """train a TSViT model according to the configuration provided"""
    set_seed(peft_config["seed"])

    train_loss=[]
    eval_loss=[]
    eval_dice=[]
    semisupervised=peft_config["semisupervised"] if "semisupervised" in    peft_config.keys() else False
    if semisupervised:
        net_list=[torch.load(path) for path in peft_config["net_path_list"]]
    print("loading dataset")
    

    train_dataset,eval_dataset,test_dataset=load_dataset(peft_config)
    
    print("loading the model...")
    net=load_model(peft_config)
        

    best_f1_score=0
    best_loss=10
    
    model_name=peft_config["model_name"]

    net.cuda()
    print_trainable_parameters(TSViT(TSVIT_config)),print_trainable_parameters(net)
    net=net.train()
    print("model "+model_name+" type " +MODEL_TYPE.model_type[peft_config["model_number"]]+" loaded sucessfully")
    optimizer=torch.optim.Adam(get_trainable_parameters(net),lr=peft_config["lr"]) #.parameters()
    if peft_config["add_scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3162,verbose=True)
    if semisupervised:
        lossfn1=semiSupervisedBCELoss(lambda_coef=peft_config["lambda"],net_list=net_list)
    lossfn=nn.BCEWithLogitsLoss()if not ("add_dice_loss" in peft_config.keys() and peft_config["add_dice_loss"]) else DiceBCELoss() #torch.nn.CrossEntropyLoss() 
    os.makedirs(f"save_models/{model_name}")
    pickle.dump(peft_config,open(f"save_models/{model_name}/config.pkl","wb"))
    if neptune:
        run=neptune.init_run(project=run_config["neptune_project_name"],
        api_token=run_config["neptune_token"],
        name=model_name,
        custom_run_id=model_name)
        hyperparams={
                "learning rate":peft_config["lr"],
                "scheduler drop":"radical 10 per 2 epochs" if peft_config["add_scheduler"] else "constant",
                "model name":model_name,
                "technique":MODEL_TYPE.model_type[peft_config["model_number"]],
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
        loader=tqdm(DataLoader(train_dataset,batch_size=32, num_workers=2,shuffle=True))
        for X,y,m in loader:
            optimizer.zero_grad()
            X=X.permute(0,1,3,4,2).to(torch.float).to("cuda")
            yp=net(X)
            yp=yp.squeeze(1)
            if semisupervised:
                loss=lossfn1(X,yp,y.cuda(),m.cuda())
                ym=y.sum(axis=1)!=0
                y=y[:,2]==0
                y=y[ym]
                yp=yp[ym]
                mask=(m[:,0]!=0)
                mask=mask[ym]
                yp=yp[mask]
                y=y[mask]
            else:
                    ym=y.sum(axis=1)!=0
                    y=y[:,2]==0
                    y=y[ym]
                    yp=yp[ym]
                    mask=(m[:,0]!=0)
                    mask=mask[ym]
                    yp=yp[mask]
                    y=y[mask]
                    loss=lossfn(yp,y.to(torch.float).cuda())
                
            loss.backward()
            optimizer.step()
            train_loss.append(loss_and_metrics_single_class(y.cpu().detach().to(torch.float).numpy(),yp.to(torch.float).cpu().detach().numpy()))

            
            loader.set_description(str(train_loss[-1]))
        if neptune:
                train_loss=np.array(train_loss).mean(axis=0)
                run["train/batch/loss"].log(train_loss[0])
                run["train/batch/acc"].log(train_loss[2])
                run["train/batch/f1score"].log(train_loss[1])
                run["train/batch/kappa"].log(train_loss[5])
        temp_eval_loss=[]
        if epoch==peft_config["number_of_epochs"]-1:
                torch.save(net.state_dict(),f"save_models/{model_name}/epoch_{epoch}.pt")
        with torch.no_grad():
            net.eval()
            if (peft_config["test_and_eval_split"]):
                dice=F1score()
                for X,y,mask in tqdm(DataLoader(eval_dataset,batch_size=64,num_workers=2)):
                    with torch.no_grad():
                            X=X.permute(0,1,3,4,2).to(torch.float).cuda()
                            yp=net(X)
                            yp=yp.squeeze()
                            y=(y[:,0]!=0)
                            mask=mask[:,0]!=0
                            yp=yp[mask]
                            y=y[mask].cuda()
                            dice(yp.cpu(),y.cpu())
                            loss=lossfn(yp,y.to(torch.float).cuda())
                            temp_eval_loss.append(loss.item())
                eval_dice.append(dice.Macro_f1())
                print("eval dice=", eval_dice[-1])
                if(best_f1_score<eval_dice[-1]):
                        torch.save(net.state_dict(),f"save_models/{model_name}/best_f1score.pt")
                        best_f1_score=eval_dice[-1]
                if neptune:
                    run["validation/epoch/f1score"].log(eval_dice[-1])
            elif not peft_config["no_eval"]:
                for X,y,m in DataLoader(eval_dataset,batch_size=64, num_workers=16):
                    optimizer.zero_grad()
                    X=X.permute(0,1,3,4,2).to(torch.float).to("cuda")#torch.tensor(X)
                    yp=net(X)
                    ym=y.sum(axis=1)!=0
                    y=y[:,2]==0
                    y=y[ym]
                    yp=yp.permute(0,2,3,1)[ym]
                    mask=(m[:,0]!=0)
                    mask=mask[ym]
                    yp=yp[mask]
                    y=y[mask]
                    loss=lossfn(yp,y.to(torch.long).cuda())
                    temp_eval_loss.append(loss.item())
            eval_loss.append(np.array(temp_eval_loss).mean())
            print("eval loss= ",eval_loss[-1])
            print("end of epoch",epoch)
            if(best_loss>eval_loss[-1]):
                        best_loss=eval_loss[-1]
                        
                        torch.save(net.state_dict(),f"save_models/{model_name}/best_loss.pt")
                        
            if neptune:
                run["validation/epoch/loss"].log(eval_loss[-1])
            net.train()
            if peft_config["add_scheduler"]:
                        scheduler.step()
                        
    net.eval()
    print("starting tresting")
    
    test_loss_last,test_dice_last =test(net,test_dataset,lossfn)
    run["test/last_epoch_loss"]=test_loss_last
    run["test/last_epoch_dice"]=test_dice_last

    net.load_state_dict(torch.load(f"save_models/{model_name}/best_f1score.pt"))
    test_loss_best_f1,test_dice_best_f1 =test(net,test_dataset,lossfn)
    run["test/best_f1_loss"]=test_loss_best_f1
    run["test/best_f1_dice"]=test_dice_best_f1

    net.load_state_dict(torch.load(f"save_models/{model_name}/best_loss.pt"))
    test_loss_best_f1,test_dice_best_f1 =test(net,test_dataset,lossfn)
    run["test/best_loss_loss"]=test_loss_best_f1
    run["test/best_loss_dice"]=test_dice_best_f1

    if neptune:
        run.stop()



if __name__=="__main__":

    my_TSVIT_config={
        'img_res':24,
        'patch_size':2,
        'num_classes':1,
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
    
    from neptune_config import NEPTUNE_API_TOKEN,PROJECT_NAME
    run_config={
        "datalist_path":"D:\\GEOAI\\code\\crop-monitoring-TSViT\\peft\\data.pkl" ,
        "model_number":MODEL_TYPE.ADAPTER_TUNE,
        "data_path":"D:\\GEOAI\\code\\Requested_Tiffs_lcc\\cropped_tiffs_24",
        "model_name":"bla_bla_blas",
        "initial_wait_file":"D:\\GEOAI\\code\\crop-monitoring-TSViT\\peft\\model.pt",
        "add_dice_loss":False,
        "lr":1e-4,
        # "r":1,
        # "rs":1,
        # "rt":2,
        "number_of_epochs":20,
        # "external":True,
        "seed":22,  
        # "temporal_prompt_dim":0,
        # "spatial_prompt_dim":8,
        "temporal_adapter_dim":8,
        "spatial_adapter_dim":8,
        "test_and_eval_split":True,
        "no_eval":False,
        "do_neptune":True,
        "add_scheduler":False,
        "vertical_flip":True,
        "horizontal_flip":True,
        "semisupervised":False,
        "my_TSViT_config":my_TSVIT_config,

        "neptune_project_name":PROJECT_NAME,
        "neptune_token":NEPTUNE_API_TOKEN,
    }
    PEFTTrain(run_config)

    
