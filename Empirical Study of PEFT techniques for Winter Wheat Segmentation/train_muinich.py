import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import numpy as np

import os
import neptune
import pickle
from tqdm import tqdm

from models.model_loader import create_model,MODEL_TYPE
from utils.util import print_trainable_parameters,get_trainable_parameters,loss_and_metrics_multi_class,set_seed
from utils.loss import DiceBCELoss

def test(net,dataset):
        """
        This code runs the test
        return the mean of the loss and metric function for all batches
        """
        temp_test_loss=[]
        with torch.no_grad():
            net.eval()
            for X,y in tqdm(DataLoader(dataset,batch_size=16,num_workers=2)):
                with torch.no_grad():
                    X,y=X.cuda(),y.cuda()
                    yp=net(X)
                    temp_test_loss.append(loss_and_metrics_multi_class(y.cpu().detach().to(torch.float).numpy(),yp.to(torch.float).cpu().detach().numpy(),criterion=torch.nn.CrossEntropyLoss()))

            test_loss=np.array(temp_test_loss).mean(axis=0)
            print("test loss= ",test_loss)
            return test_loss
         


def PEFTTrain ( peft_config):

    set_seed(peft_config["seed"])#set seed to reproduce the results

    peft_config["technique"]=MODEL_TYPE().to_string(peft_config["model_number"])

    train_loss=[]
    eval_loss=[]

    #loading datasets 
    train_dataset=peft_config["train_dataset"] 
    eval_dataset=peft_config["eval_dataset"] 
    test_dataset=peft_config["test_dataset"] 
    peft_config["train_dataset"]=str(peft_config["train_dataset"])
    peft_config["eval_dataset"]=str(peft_config["eval_dataset"])
    peft_config["test_dataset"]=str(peft_config["test_dataset"])
    

    print("loading the model:")
    net=create_model(peft_config)
    
    #initiallizing the parameters used to save the best scores for eval
    best_f1score=0
    best_iou=0
    best_loss=10
    
    model_name=peft_config["model_name"]

    net.cuda()
    # print_trainable_parameters(create_model(TSVIT_config)),
    print_trainable_parameters(net)

    optimizer=torch.optim.Adam(get_trainable_parameters(net),lr=peft_config["lr"]) 

    if peft_config["add_scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3162,verbose=True)
    
    lossfn=nn.CrossEntropyLoss() if not ("add_dice_loss" in peft_config.keys() and peft_config["add_dice_loss"]) else DiceBCELoss()  
    
    #creating a directory for the model, where weight will be save, also, saving the configurations
    os.makedirs(f"save_models/{model_name}")
    pickle.dump(peft_config,open(f"save_models/{model_name}/config.pkl","wb"))
    
    #if neptune is True, initialize the connection and load the configuration
    if peft_config["do_neptune"]:
        run=neptune.init_run(project=peft_config["project_name"],
        api_token=peft_config["api_token"],
        name=model_name,
        custom_run_id=model_name)
        run["configuration"] = peft_config
    
    
    print("starting training")
    
    for epoch in range(peft_config["number_of_epochs"]):
        net.train()
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
        if peft_config["do_neptune"]:        
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
            
            eval_loss.append(np.array(temp_eval_loss).mean(axis=0))
            print("eval",epoch,eval_loss[-1])
            if peft_config["do_neptune"]:        
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
                        

            if peft_config["add_scheduler"]:
                        scheduler.step()
                        
    net.eval()


    print("starting testing")
    print("testing model  final epoch")
    test_loss_last =test(net,test_dataset)
    if peft_config["do_neptune"]:
        run["test/last_epoch/loss"]=(test_loss_last[0])
        run["test/last_epoch/acc"]=(test_loss_last[1])
        run["test/last_epoch/iou"]=(test_loss_last[2])
        run["test/last_epoch/f1score"]=(test_loss_last[3])


    print("testing model with best eval f1-score")
    net.load_state_dict(torch.load(f"save_models/{model_name}/best_f1score.pt"))
    
    test_loss_best_f1 =test(net,test_dataset) 
    if peft_config["do_neptune"]:
        run["test/best_f1/loss"]=(test_loss_best_f1[0])
        run["test/best_f1/acc"]=(test_loss_best_f1[1])
        run["test/best_f1/iou"]=(test_loss_best_f1[2])
        run["test/best_f1/f1score"]=(test_loss_best_f1[3])

    print("testing model with best eval loss")
    net.load_state_dict(torch.load(f"save_models/{model_name}/best_loss.pt"))
    test_loss_best_f1 =test(net,test_dataset)
    if peft_config["do_neptune"]:
        run["test/best_loss/loss"]=(test_loss_best_f1[0])
        run["test/best_loss/acc"]=(test_loss_best_f1[1])
        run["test/best_loss/iou"]=(test_loss_best_f1[2])
        run["test/best_loss/f1score"]=(test_loss_best_f1[3])


    if peft_config["do_neptune"]:
        run.stop()



if __name__=="__main__":
    my_TSViT_config={
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
    
    from datasets_class.munich_dataset import munich_dataset
    from config import NEPTUNE_API_TOKEN,PROJECT_NAME,PROCCESSED_DATA_PATH
    base_path=PROCCESSED_DATA_PATH

    #get the tile id from the file name
    def get_tile(name):
        return name.split("_")[1]
    #get all tiles that have the id as on the list of ids in the original split testfile
    def get_tiles_from_split(tile_list,path):
        with open(path,"r") as f:
            ids=f.read().split("\n")
            return [a for a in tile_list if get_tile(a) in ids]
    
    #all the ids availabe in the datapath
    tile_list=[a for a in  os.listdir(base_path)]
    
    #extracting the training dataset
    train_datalist=get_tiles_from_split(tile_list,"original_dist/train_fold0.tileids")
    train_dataset=munich_dataset(base_path,train_datalist)

    #extracting the eval dataset
    eval_datalist=get_tiles_from_split(tile_list,"original_dist/eval.tileids")
    eval_dataset=munich_dataset(base_path,eval_datalist)
    
    #extracting the test dataset
    test_datalist=get_tiles_from_split(tile_list,"original_dist/test_fold0.tileids")
    test_dataset=munich_dataset(base_path,test_datalist)
    
    
    

    from models.model_loader import MODEL_TYPE
    run_config={
        #providing the dataset
        "train_dataset":train_dataset,
        "test_dataset":test_dataset,
        "eval_dataset":eval_dataset,
        
        #number of classes in munich dataset is 27
        "number_of_classes":27,

        #possible pefting techniques
        "model_number":MODEL_TYPE.LORA,


        #in adapter, lora, and promt, if true, change token, else, add head layer
        "change_to_token":True,

        #must be unique, serves as the neptune custom run id 
        "model_name":"munich lora 4-4-4 1e-3",

        #initial TSViT model as provided by the model
        "initial_weight_file":"./Initial_TSViT_weights.pt",

        #general hyperparameters
        "lr":1e-3,
        "number_of_epochs":40,
        "add_scheduler":False,
        "vertical_flip":False,
        "horizontal_flip":False,
        "seed":313,

        #LORA hyperparamters
        "r":4,
        "rs":4,
        "rt":4,
        
        #Prompt Tuning hyperparameters
        "external":True,
        "temporal_prompt_dim":2,
        "spatial_prompt_dim":2,

        #Adapter Paramters
        "temporal_adapter_dim":8,
        "spatial_adapter_dim":8,

        #Token Tune Parameters
        "all_tokens":True, #in token extended, if true train all tokens, else, train 8 tokens (27 from munich -  19 from PASTIS)
        "model_config":my_TSViT_config,
        "full":True,#if full = true, train all paramters else, train token only
        
        
        #Neptune settings 
        "do_neptune":False,
        "project_name":PROJECT_NAME, #fix in config
        "api_token":NEPTUNE_API_TOKEN , #fix in config
    }
    PEFTTrain(run_config)

  