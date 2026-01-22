import os
import time
import sys
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
from compressai.zoo import load_state_dict  
from compressai.models.stf import SymmetricalTransFormer as STF
import matplotlib.pyplot as plt
sys.path.append('/home/mmproj01/repos/ALICE')
from utils.LoadModel import Ready_Model
from .plot import plot_psnr_bpp
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def collect_images(rootpath: str):
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

@torch.no_grad()
def inference(model, x: torch.Tensor):
    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)

    p = 64
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    pl = (new_w - w) // 2
    pr = new_w - w - pl
    pt = (new_h - h) // 2
    pb = new_h - h - pt

    x_padded = F.pad(x, (pl, pr, pt, pb), mode="constant", value=0)

    t0 = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - t0

    t1 = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - t1

    x_hat = out_dec["x_hat"]
    x_hat = F.pad(x_hat, (-pl, -pr, -pt, -pb))

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr": psnr(x, x_hat),
        "ms-ssim": ms_ssim(x, x_hat, data_range=1.0).item(),
        "bpp": float(bpp),
        "encoding_time": float(enc_time),
        "decoding_time": float(dec_time),
    }


def eval_model(model, filepaths):
    metrics = defaultdict(float)
    device = next(model.parameters()).device

    for f in filepaths:
        x = read_image(f).to(device)
        rv = inference(model, x)
        for k, v in rv.items():
            metrics[k] += float(v)

    for k in metrics:
        metrics[k] /= len(filepaths)

    return dict(metrics)

def main(model=None,rate=0.0483):  
    dataset = Path("~/datasets/kodak").expanduser()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset}")
    filepaths = collect_images(dataset)
    if not filepaths:
        raise SystemExit("No images found in dataset folder.")
    
    if model is not None:
        # Evaluate the provided model
        metrics = eval_model(model, filepaths)
        print(metrics)
        results = {rate: metrics}
    else:
        runs = [0.0018,0.0035,0.0067,0.0130,0.0250,0.0483]
        results = {}
        for lam in sorted(runs):
            model = Ready_Model(lam)
            metrics = eval_model(model,filepaths)
            results[lam] = metrics
            print(f"Rate {lam}: {metrics}")
    lams = sorted(results.keys())
    json_out = {
        "dataset": dataset.name,
        "model": "STF",
        "lambdas": lams,
        "metrics": {
            "psnr": [results[lam]["psnr"] for lam in lams],
            "ms-ssim": [results[lam]["ms-ssim"] for lam in lams],
            "bpp": [results[lam]["bpp"] for lam in lams],
            "encoding_time": [results[lam]["encoding_time"] for lam in lams],
            "decoding_time": [results[lam]["decoding_time"] for lam in lams],
        },
    }
    results_dir = os.path.join('/home/mmproj01/repos/ALICE', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    out_json = os.path.join(results_dir, "metrics_results.json")
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\nSaved: {out_json}")

    # plot_psnr_bpp()

if __name__ == "__main__":
    main()
