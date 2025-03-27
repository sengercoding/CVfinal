import torch
from src.model import generator
from torch.autograd import Variable
from scipy.ndimage import zoom
import cv2
import os
from PIL import Image
import argparse
import numpy as np
from skimage.color import rgb2yuv,yuv2rgb,rgb2gray
import matplotlib.pyplot as plt



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
    
def resize_bw(img, output_size=(256, 256)):
    zoom_factors = (output_size[0] / img.shape[0], output_size[1] / img.shape[1], 1)
    return zoom(img, zoom_factors, order=1) 

def resize_rgb(img, output_size=(256, 256)):
    h, w, c = img.shape
    zoom_factors = (output_size[0] / h, output_size[1] / w, 1)
    resized = zoom(img, zoom_factors, order=1)
    return resized.astype(np.uint8)


    
def colorize(G, img_bw):
    # assert img_bw.ndim == 2, "black and white image array needs to be two dimensional"
    # if img_bw.shape != (256,256):
    #     img_bw = resize_bw(img_bw)
    img_bw = Image.fromarray(img_bw).convert("RGB")
    img_yuv = rgb2yuv(img_bw)
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
    rgb = yuv2rgb(yuv.transpose(1,2,0))
    return rgb

def visualize(G, img):
    """
    Visualizes side-by-side the ground truth, grayscale and colorized image
    """
    if img.shape != (256,256,3):
        img = resize_rgb(img)
        print("image has been resized!")
    img_bw = (rgb2gray(img)*255).astype(np.uint8)
    img_col = colorize(G,img_bw)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,5))
    ax1.imshow(img)
    ax1.set_title("ground truth")
    ax2.imshow(img_bw, cmap="gray")
    ax2.set_title("grayscale")
    ax3.imshow(img_col)
    ax3.set_title("colorized")
    for ax in [ax1,ax2,ax3]:
        ax.set_axis_off()
    plt.show()


def load_generator():
    G = generator()
    G.load_state_dict(torch.load("src/G.pth",map_location=torch.device('cpu')))
    print("model has loaded sucessfully!")
    return G