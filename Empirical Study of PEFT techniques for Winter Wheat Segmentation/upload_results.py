from neptune.types import File
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import neptune
from PIL import Image

@torch.no_grad()
def upload(dataset,run,net,number_of_images):
    i=0
    dataset.set_upload_images(True)
    torch.random.manual_seed(22)#
    
    for image,X,y,m in DataLoader(dataset,1,shuffle=True):
        i+=1

        for t in range(9):
            print(X[0,t,0:3,:,:].permute(1,2,0).shape)
            img=image[0,t,0:3,:,:].permute(1,2,0).numpy()
            img=np.flip(img,axis=2)
            img=img/img.max(axis=0, keepdims=True)
            img=img*255
            img=img.astype(np.uint8)
            img=Image.fromarray(img)
            run["results/image"+str(i)].append(img)

        yp=net(torch.tensor(X).permute(0,1,3,4,2).to(torch.float).cuda()).cpu().numpy()[0]
        image=np.zeros((24,24,3),dtype=np.uint8)
        print(yp.max(),yp.min())
        yp=yp>0
        
        yp=yp[0]
        y= y[:,0]!=0
        y=y[0].numpy()
        m=(m[0,0]!=0).numpy()
        print(yp.dtype,y.dtype,m.dtype)
        print((y&yp&m).dtype)
        image[y&yp&m]=(0,255,0)
        image[~y&~yp&m]=(255,255,255)
        image[~y&yp&m]=(255,0,0)
        image[y&~yp&m]=(0,0,255)
        print(image[y&yp&m].shape)
        # print(image)
        # break
        run["results/image"+str(i)].append(Image.fromarray(image.astype(np.uint8)))
        if i==number_of_images:
            break
if __name__=="__main__":
    from loraTSViT import LoraTSViT
    from dataset import TSViT_leb_dataset
    
    TSViT_config={
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
    
    model_name="LoraTest r2 lr 1e-3 flip 2"
    model_path="save_models/"+model_name+"/best_f1score.pt"
    
    net=LoraTSViT(TSViT_config,r=2)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    test_dataset=TSViT_leb_dataset("../data/cropped_tiffs_24",[2020],aois=[0,1,2,3,4])
    
    from neptune_config import NEPTUNE_API_TOKEN,PROJECT_NAME

    run=neptune.init_run(project=PROJECT_NAME,
    api_token=NEPTUNE_API_TOKEN,
    name="test images 1223121321321321",
    custom_run_id="test images 7")
    upload(test_dataset,run,net,10)