import pickle
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
import os
import neptune

from datasets_class.lebanon_dataset import TSViT_leb_dataset
from models.TSViT import TSViT
from utils.util import print_trainable_parameters,get_trainable_parameters,loss_and_metrics_single_class,F1score,set_seed
from models.model_loader import TSVIT_config,create_model,MODEL_TYPE


def load_dataset(config):
    datalist=None
    if "datalist_path" in config.keys():
        with open(config["datalist_path"],'rb') as f:
            datalist=pickle.load(f)
    
        # if "train_years" in config.keys():
        #         # print(config["train_years"])
        #         datalist=[data for data in datalist if data[1] in config["train_years"]]
            
    data_path=config["data_path"]

    vertical_flip=config["vertical_flip"] if "vertical_flip" in config else False
    horizontal_flip=config["horizontal_flip"] if "horizontal_flip" in config else False

    train_min_pixels_per_image=config["train_min_pixels_per_image"] if "train_min_pixels_per_image" in config.keys() else 1
    eval_min_pixels_per_image=config["eval_min_pixels_per_image"] if "eval_min_pixels_per_image" in config.keys() else 1
    test_min_pixels_per_image=config["test_min_pixels_per_image"] if "test_min_pixels_per_image" in config.keys() else -1

    train_dataset=TSViT_leb_dataset(basepath=data_path,
                                    list_of_tile=datalist,
                                    min_pixels_per_image=train_min_pixels_per_image,
                                    years=config["train_years"],
                                    aois=config["train_aois"],
                                    vertical_flip=vertical_flip,
                                    horizontal_flip=horizontal_flip,
                                    upload_image=False)
    
    eval_dataset=TSViT_leb_dataset(basepath=data_path,
                                    list_of_tile=datalist,
                                    min_pixels_per_image=eval_min_pixels_per_image,
                                    years=config["eval_years"],
                                    aois=config["eval_aois"],
                                    vertical_flip=False,
                                    horizontal_flip=False,
                                    upload_image=False)
    test_dataset=TSViT_leb_dataset(basepath=data_path,
                                    list_of_tile=datalist,
                                    min_pixels_per_image=test_min_pixels_per_image,
                                    years=config["test_years"],
                                    aois=config["test_aois"],
                                    vertical_flip=False,
                                    horizontal_flip=False,
                                    upload_image=False)
    return train_dataset,eval_dataset,test_dataset

def test(net,dataset,lossfn):
        temp_eval_loss=[]
        with torch.no_grad():
            net.eval()
            dice=F1score()
            for X,y,mask in tqdm(DataLoader(dataset,batch_size=64,num_workers=1)):
                with torch.no_grad():
                        X=X.permute(0,1,3,4,2).to(torch.float).cuda()
                        yp=net(X)
                        yp=yp.squeeze()
                        # y=(y[:,0]!=0)
                        # mask=mask[:,0]!=0
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

    eval_loss=[]
    eval_dice=[]

    print("loading dataset")
    train_dataset,eval_dataset,test_dataset=load_dataset(peft_config)
    
    print("loading the model...")
    net=create_model(peft_config)
        

    best_f1_score=0
    best_loss=10
    net.cuda()

    model_name=peft_config["model_name"]

    
    print_trainable_parameters(TSViT(TSVIT_config))
    print_trainable_parameters(net)
    print("model "+model_name+" type " +MODEL_TYPE().to_string(peft_config["model_number"])+" loaded sucessfully")
    
    optimizer=torch.optim.Adam(get_trainable_parameters(net),lr=peft_config["lr"])
    if peft_config["add_scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3162,verbose=True)
    
    lossfn=nn.BCEWithLogitsLoss() 

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
                "technique":MODEL_TYPE().to_string(peft_config["model_number"]),
                "trainable parameters": print_trainable_parameters(net)[0],
                "total number of parameters":print_trainable_parameters(net)[1],
                "percentange of trainable parameters": print_trainable_parameters(net)[2],
                "additional parameters":print_trainable_parameters(net)[1]-print_trainable_parameters(TSViT(TSVIT_config))[1],
        }
        run["hyperparameters"] = hyperparams
        run["configuration"] = peft_config

    print("starting training")
    for epoch in range(peft_config["number_of_epochs"]):
        train_loss=[]
        net.train()
        loader=tqdm(DataLoader(train_dataset,batch_size=32, num_workers=2,shuffle=True))
        for X,y,m in loader:
            optimizer.zero_grad()
            X=X.permute(0,1,3,4,2).to(torch.float).to("cuda")
            yp=net(X)
            yp=yp.squeeze(1)
            # ym=y.sum(axis=1)!=0
            # y=y[:,2]==0
            # y=y[ym]
            # yp=yp[ym]
            # mask=(m[:,0]!=0)
            # mask=mask[ym]
            yp=yp[m]#ask
            y=y[m]#ask
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
            # if (peft_config["test_and_eval_split"]):
            dice=F1score()
            for X,y,mask in tqdm(DataLoader(eval_dataset,batch_size=64,num_workers=2)):
                with torch.no_grad():
                        X=X.permute(0,1,3,4,2).to(torch.float).cuda()
                        yp=net(X)
                        yp=yp.squeeze()
                        # y=(y[:,0]!=0)
                        # mask=mask[:,0]!=0
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

            # for X,y,m in DataLoader(eval_dataset,batch_size=64, num_workers=2):
            #     optimizer.zero_grad()
            #     X=X.permute(0,1,3,4,2).to(torch.float).to("cuda")#torch.tensor(X)
            #     yp=net(X)
            #     # ym=y.sum(axis=1)!=0
            #     # y=y[:,2]==0
            #     # y=y[ym]
            #     # yp=yp.permute(0,2,3,1)[ym]
            #     # mask=(m[:,0]!=0)
            #     # mask=mask[ym]
            #     yp=yp[m]#ask
            #     y=y[m]#mask
            #     loss=lossfn(yp,y.to(torch.long).cuda())
            #     temp_eval_loss.append(loss.item())
            eval_loss.append(np.array(temp_eval_loss).mean())
            print("eval loss= ",eval_loss[-1])
            print("end of epoch",epoch)
            if(best_loss>eval_loss[-1]):
                        best_loss=eval_loss[-1]
                        
                        torch.save(net.state_dict(),f"save_models/{model_name}/best_loss.pt")
                        
            if neptune:
                run["validation/epoch/loss"].log(eval_loss[-1])
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
        'num_classes':16,
        "max_seq_len":60,
        'dim':128,
        'temporal_depth':8,
        'spatial_depth': 8,
        'heads': 4,
        'pool': 'cls',
        'dim_head': 32,
        'emb_dropout': 0.5,
        'scale_dim': 4,
        'dropout':0,
        'num_channels': 11,
        'num_feature':16,
        'scale_dim':4,
        'ignore_background': False
    }
    
    from config import NEPTUNE_API_TOKEN,PROJECT_NAME
    run_config={
          
          #general parameters
        "model_name":"lebanon test 2",#must be unique
        "number_of_classes":1,#only wheat
        "lr":1e-4,
        "seed":22, 
        "change_to_token":False,#for Lebanon, it is recommended to keep false
        "model_number":MODEL_TYPE.RANDOM_FTSVIT,
        "number_of_epochs":20,
        # "test_and_eval_split":True,
        "add_scheduler":False,
        "vertical_flip":True,
        "horizontal_flip":True,
        "my_TSViT_config":my_TSVIT_config,


        "model_config":my_TSVIT_config,

        "train_years":[2016,2017,2018,2019,2020],
        "train_aois":[0,1,2,3,4],
        "train_min_pixels_per_image":1,

        "eval_years":[2020],
        "eval_aois":[0],
        "eval_min_pixels_per_image":0,

        "test_years":[2020],
        "test_aois":[0,1,2,3,4],
        "test_min_pixels_per_image":-1,

        #TODO change after preprocessing
        "datalist_path":"lbdata.pkl" ,
        "initial_weight_file":"Initial_TSViT_weights.pt",
        "data_path":"D:\\GEOAI\\code\\Requested_Tiffs_lcc\\cropped_tiffs_24",

        #LORA hyperparamters
        "r":1,
        "rs":1,
        "rt":2,

        #Prompt Hyperparamters
        "external":True,
        "temporal_prompt_dim":0,
        "spatial_prompt_dim":8,
        
        #Adapter HyperParamters
        "temporal_adapter_dim":8,
        "spatial_adapter_dim":8,
        
        #Neptune Config        
        "do_neptune":True,
        "neptune_project_name":PROJECT_NAME,
        "neptune_token":NEPTUNE_API_TOKEN,
    }
    PEFTTrain(run_config)

    

