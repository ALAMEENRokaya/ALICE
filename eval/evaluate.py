import torch
import torchvision.transforms as T
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def PSNR(a,b):
    mse=F.mse_loss(a,b).item()
    return -10*math.log10(mse)

def get_image(path):
    img=Image.open(path).convert('RGB')
    transform=T.ToTensor()
    img=transform(img).unsqueeze(0)
    return img

def inference(model,x,reconstruction_path):
    torch.no_grad()
    # x should be changed to fit expected input of model
    start = time.time()
    compressed=model.compress(x)
    enc_time=time.time()-start
    decompressed=model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time=time.time()-start
    # reconstruct + bpp 
    # we should be able to return this{
    #     "psnr":,
    #     "ms-ssim": ,
    #     "bpp": bpp,
    #     "encoding_time": enc_time,
    #     "decoding_time": dec_time,
    # }
