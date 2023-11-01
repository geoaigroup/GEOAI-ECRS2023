import train_muinich
import os
from models.model_loader import MODEL_TYPE

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
    
    from munich_dataset import munich_dataset
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
        "do_neptune":True,
        "project_name":PROJECT_NAME, #fix in config
        "api_token":NEPTUNE_API_TOKEN , #fix in config
    }
    train_muinich.PEFTTrain(run_config)