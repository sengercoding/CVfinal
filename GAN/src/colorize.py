import torch
from src.model import generator
from torch.autograd import Variable
from scipy.ndimage import zoom
import cv2
import os
from PIL import Image
import argparse
import numpy as np
from skimage.color import rgb2yuv,yuv2rgb



def inference(G,in_path,out_path = None):
    p=Image.open(in_path).convert('RGB')
    img_yuv = rgb2yuv(p)
    H,W,_ = img_yuv.shape
    infimg = np.expand_dims(np.expand_dims(img_yuv[...,0], axis=0), axis=0)
    img_variable = Variable(torch.Tensor(infimg-0.5))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda(torch.cuda.current_device())
    res = G(img_variable)
    uv=res.cpu().detach().numpy()
    uv[:,0,:,:] *= 0.436
    uv[:,1,:,:] *= 0.615
    (_,_,H1,W1) = uv.shape
    uv = zoom(uv,(1,1,H/H1,W/W1))
    yuv = np.concatenate([infimg,uv],axis=1)[0]
    rgb=yuv2rgb(yuv.transpose(1,2,0))
    if out_path is not None:
        cv2.imwrite(out_path,(rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])
    else:
        return rgb
    
def infer_dir(G, in_dir, out_dir):
    """ 
    colorizes a whole directory
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_path)
        print(f"created new directory {out_dir}")

    for img in os.listdir(in_dir):
        file_name = os.fsdecode(img)

        if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
            out_path = os.path.join(out_dir, file_name, "_col")
            inference(G, img, out_path=out_path)
            
        else:
            raise Warning(f"skipped file {file_name}, since it is not a jpg or jpeg")



def load_model():
    G = generator()
    G.load_state_dict(torch.load("src/model.pth",map_location=torch.device('cpu')))
    print("model has loaded sucessfully!")
    return G

