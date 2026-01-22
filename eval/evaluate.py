import os
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
from compressai.zoo import load_state_dict  
from compressai.models.stf import SymmetricalTransFormer as STF
import matplotlib.pyplot as plt
from plot import plot_psnr_bpp

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


def main():

    dataset = Path("~/datasets/kodak").expanduser()
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

    if not dataset.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset}")
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Checkpoint folder not found: {ckpt_dir}")


    filepaths = collect_images(dataset)
    if not filepaths:
        raise SystemExit("No images found in dataset folder.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for lam in sorted(runs.keys()):
        ckpt_path = os.path.join(ckpt_dir, runs[lam])
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"Missing checkpoint file: {ckpt_path}")
        print(f"\nEvaluating λ={lam}  ({ckpt_path})")
        model = load_stf_checkpoint(ckpt_path, device)
        metrics = eval_model(model, filepaths)
        results[lam] = metrics
        print(metrics)
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

    out_json = "metrics_results.json"
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\nSaved: {out_json}")
    plot_psnr_bpp()

    


if __name__ == "__main__":
    main()
