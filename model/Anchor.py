import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import time
import math
import json
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import compressai
import torch.nn as nn
from compressai.zoo import load_state_dict  
from compressai.models.stf import SymmetricalTransFormer as STF
import matplotlib.pyplot as plt
sys.path.append('/home/mmproj01/repos/ALICE')
from utils.FCWrapper import wrap_fc_layers
from utils.LoadModel import Ready_Model
from eval.evaluate import main as eval_main

model=Ready_Model(0.0483)
model = wrap_fc_layers(model)
metrics=eval_main(model,0.0483)
print(metrics)
