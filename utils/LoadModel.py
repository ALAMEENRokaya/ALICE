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


def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            return ckpt_obj[k]
    return ckpt_obj


def _strip_prefix(state, prefix: str):
    if any(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_stf_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    state = _strip_prefix(state, "net.")

    try:
        state = load_state_dict(state)
    except Exception:
        pass

    model = STF(
        pretrain_img_size=256,
        patch_size=2,
        in_chans=3,
        embed_dim=48,
        window_size=4,
    )

    missing, unexpected = torch.nn.Module.load_state_dict(model, state, strict=False)

    if hasattr(model, "update"):
        try:
            model.update(force=True)
        except TypeError:
            model.update()

    model = model.to(device).eval()

    if len(missing) > 50 or len(unexpected) > 50:
        print("\n[ERROR] Checkpoint/model mismatch:", checkpoint_path)
        print("Missing:", len(missing), "Unexpected:", len(unexpected))
        print("Missing sample:", missing[:20])
        print("Unexpected sample:", unexpected[:20])
        raise SystemExit(
            "Your STF constructor args (embed_dim/window_size/etc.) likely do NOT match the checkpoint.\n"
            "If these are official STF checkpoints, we need the exact STF config used in training."
        )

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
    compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
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