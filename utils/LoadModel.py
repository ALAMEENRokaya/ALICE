import os
import time
import math
from collections import defaultdict
from pathlib import Path
import torch
import compressai
from compressai.zoo import load_state_dict
from compressai.models.stf import SymmetricalTransFormer as STF
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.Masked import wrapmask_fc_layers

# Entropy buffers are recomputed by model.update(); strip them before loading
_ENTROPY_BUFFERS = {
    "entropy_bottleneck._offset",
    "entropy_bottleneck._quantized_cdf",
    "entropy_bottleneck._cdf_length",
    "gaussian_conditional._offset",
    "gaussian_conditional._quantized_cdf",
    "gaussian_conditional._cdf_length",
    "gaussian_conditional.scale_table",
}

def load_stf_checkpoint(path: str, device: torch.device) -> "STF":
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # remove DDP/DataParallel prefix
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    model = STF(
        pretrain_img_size=256,
        patch_size=2,
        in_chans=3,
        embed_dim=48,
        window_size=4,
    )

    # detect masked+adapter architecture (batata checkpoints)
    is_masked = any("mlp.fc1.mask" in k for k in state.keys())
    if is_masked:
        model = wrapmask_fc_layers(model, alpha=8, r=8, num_bitrates=6)

    # entropy buffers (shape depends on trained quantiles) are recomputed by update()
    state = {k: v for k, v in state.items() if k not in _ENTROPY_BUFFERS}
    torch.nn.Module.load_state_dict(model, state, strict=False)

    # IMPORTANT: initialize entropy model CDF tables
    model.update(force=True)
    model = model.to(device).eval()

    return model

def Ready_Model(rate):  
    ckpt_dir = Path("~/checkpoints").expanduser()
    runs = {
            0.0018: "stf_0018.pth.tar",
            0.0035: "stf_0035.pth.tar",
            0.0067: "stf_0067.pth.tar",
            0.0130: "stf_013.pth.tar",
            0.0250: "stf_025.pth.tar",
            0.0483: "stf_0483.pth.tar",
        }    
    
    #compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Checkpoint folder not found: {ckpt_dir}")
    if rate not in runs:
        raise SystemExit(f"Rate {rate} not found in runs dict.")
    
    ckpt_filename = runs[rate]
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Missing checkpoint file: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_stf_checkpoint(ckpt_path, device)
    return model 


def Ready_Models(rates):
    return [Ready_Model(rate) for rate in rates]