import math
import os
import sys
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
import wandb
import json
import matplotlib.pyplot as plt

from compressai.datasets.utils import ImageFolder

sys.path.append('/home/mmproj01/repos/ALICE')
from utils.LoadModel import Ready_Model
from utils.Maksadapters import (
    apply_mask_to_all_linear,
    add_adapters_to_mlp,
    BaseMaskedFC,
    BaseMaskedWithAdapterFC,
)
from eval.evaluate import collect_images, eval_model

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
LMBDA = 0.025
BITRATE_IDX = 0   # single-row mask, always index 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def set_bitrate(model, idx: int):
    for m in model.modules():
        if hasattr(m, "set_bitrate") and callable(m.set_bitrate):
            m.set_bitrate(idx)


def gate_sparsity(model):
    ons = []
    for m in model.modules():
        if hasattr(m, "mask") and hasattr(m, "bitrate_idx"):
            g = (m.mask[m.bitrate_idx] > 0).float()
            ons.append(g.mean().item())
    return sum(ons) / len(ons) if ons else None


def load_baseline_curve(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    baseline_psnr = data["psnr"]
    baseline_bpp  = data["bpp"]
    assert len(baseline_psnr) == len(baseline_bpp), "psnr and bpp must have same length"
    return baseline_bpp, baseline_psnr


def wandb_log_psnr_bpp_progress(baseline_bpp, baseline_psnr, epoch_points, epoch):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(baseline_bpp, baseline_psnr, marker="o")
    xs = [p["bpp"]  for p in epoch_points]
    ys = [p["psnr"] for p in epoch_points]
    ax.scatter(xs, ys)
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR vs BPP (epoch {epoch})")
    wandb.log({"plots/psnr_vs_bpp_progress": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth.tar"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, os.path.join(out_dir, "best.pth.tar"))


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
class RateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"]     = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out


# ------------------------------------------------------------------
# Train / validate
# ------------------------------------------------------------------
def train(model, train_loader, optimizer, loss_function, device, max_iters=20_000):
    model.train()
    set_bitrate(model, BITRATE_IDX)
    running = {"loss": 0, "mse": 0, "bpp": 0}
    iters = 0
    pbar = tqdm(total=max_iters, desc="Training", leave=False)
    while iters < max_iters:
        for inputs in train_loader:
            targets = inputs
            inputs  = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs  = model(inputs)
            loss_out = loss_function(outputs, targets, LMBDA)
            loss_out["loss"].backward()
            optimizer.step()
            running["loss"] += loss_out["loss"].item()
            running["mse"]  += loss_out["mse_loss"].item()
            running["bpp"]  += loss_out["bpp_loss"].item()
            iters += 1
            pbar.update(1)
            if iters >= max_iters:
                break
    pbar.close()
    for k in running:
        running[k] /= iters
    return running


def validate(model, val_loader, loss_function, device, bitrate_idx: int):
    model.eval()
    running = {"loss": 0, "mse": 0, "bpp": 0}
    n = 0
    set_bitrate(model, bitrate_idx)
    with torch.no_grad():
        for inputs in tqdm(val_loader, desc=f"Validation (idx={bitrate_idx})", leave=False):
            targets = inputs
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs  = model(inputs)
            loss_out = loss_function(outputs, targets, LMBDA)
            running["loss"] += loss_out["loss"].item()
            running["mse"]  += loss_out["mse_loss"].item()
            running["bpp"]  += loss_out["bpp_loss"].item()
            n += 1
    for k in running:
        running[k] /= n
    return running


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def train_model(model, device, train_loader, val_loader, args,
                *, filepaths, baseline_bpp, baseline_psnr, batch_size=16, epochs=13):
    best_val_loss = float("inf")
    model     = model.to(device)
    loss_fn   = RateDistortionLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=20_000)

    epoch_points = []
    for epoch in range(epochs):
        train_stats = train(model, train_loader, optimizer, loss_fn, device)
        val_stats   = validate(model, val_loader, loss_fn, device, bitrate_idx=BITRATE_IDX)

        is_best_val   = val_stats["loss"] < best_val_loss
        best_val_loss = min(best_val_loss, val_stats["loss"])

        model.eval()
        set_bitrate(model, BITRATE_IDX)
        print("idx", BITRATE_IDX, "avg gate on =", gate_sparsity(model))
        kodak_metrics = eval_model(model, filepaths)
        kodak_metrics["bitrate_idx"] = BITRATE_IDX
        kodak_metrics["lmbda"]       = LMBDA
        print(
            f"Kodak idx={BITRATE_IDX} (λ={LMBDA}) | "
            f"PSNR: {kodak_metrics['psnr']:.3f}, "
            f"MS-SSIM: {kodak_metrics['ms-ssim']:.5f}, "
            f"BPP: {kodak_metrics['bpp']:.4f}"
        )

        epoch_points.append({
            "epoch": epoch + 1,
            "bpp":  kodak_metrics["bpp"],
            "psnr": kodak_metrics["psnr"],
        })

        wandb_log_psnr_bpp_progress(baseline_bpp, baseline_psnr, epoch_points, epoch + 1)
        scheduler.step()

        save_checkpoint(
            {
                "epoch":          epoch,
                "state_dict":     model.state_dict(),
                "best_val_loss":  best_val_loss,
                "optimizer":      optimizer.state_dict(),
                "lr_scheduler":   scheduler.state_dict(),
                "args":           vars(args),
                "train_stats":    train_stats,
                "val_stats":      val_stats,
                "kodak_metrics":  kodak_metrics,
            },
            is_best_val,
            out_dir=args.save_dir,
            filename=f"checkpoint_epoch{epoch+1}.pth.tar",
        )

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train | Loss: {train_stats['loss']:.3f}, "
            f"MSE: {train_stats['mse']*255**2/3:.3f}, "
            f"BPP: {train_stats['bpp']:.3f} || "
            f"Val | Loss: {val_stats['loss']:.3f}, "
            f"MSE: {val_stats['mse']*255**2/3:.3f}, "
            f"BPP: {val_stats['bpp']:.3f}"
        )
        wandb.log({
            "epoch":               epoch + 1,
            "train/loss":          train_stats["loss"],
            "train/mse_loss":      train_stats["mse"],
            "train/bpp_loss":      train_stats["bpp"],
            "val/loss":            val_stats["loss"],
            "val/mse_loss":        val_stats["mse"],
            "val/bpp_loss":        val_stats["bpp"],
            "eval/kodak_psnr":     kodak_metrics["psnr"],
            "eval/kodak_msssim":   kodak_metrics["ms-ssim"],
            "eval/kodak_bpp":      kodak_metrics["bpp"],
            "eval/kodak_enc_time": kodak_metrics["encoding_time"],
            "eval/kodak_dec_time": kodak_metrics["decoding_time"],
            "lr":                  optimizer.param_groups[0]["lr"],
        }, step=epoch + 1)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def count_masked(model):
    return sum(1 for m in model.modules() if isinstance(m, BaseMaskedFC))


def count_linear(model):
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


# ------------------------------------------------------------------
# Arg parsing
# ------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Adapter fine-tuning at lambda=0.025.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Training dataset")
    parser.add_argument("-e", "--epochs", default=13, type=int,
                        help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--val-batch-size", type=int, default=64,
                        help="Test batch size (default: %(default)s)")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                        help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints_adapter_025",
                        help="Where to save checkpoints")
    return parser.parse_args(argv)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main(argv):
    args = parse_args(argv)
    print(args)

    wandb.init(
        project="image-compression-training",
        name=f"adapter025bs{args.batch_size}",
        config=vars(args),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*",   step_metric="epoch")
    wandb.define_metric("lr",      step_metric="epoch")

    device = torch.device("cuda:0")
    baseline_json = "/home/mmproj01/repos/ALICE/results/results.json"
    baseline_bpp, baseline_psnr = load_baseline_curve(baseline_json)
    print("Loaded baseline points:", list(zip(baseline_bpp, baseline_psnr)))

    print("Using device:", device)
    print("CUDA device name:", torch.cuda.get_device_name(0))

    kodak_dir = Path("~/datasets/kodak").expanduser()
    filepaths = collect_images(str(kodak_dir))
    if not filepaths:
        raise SystemExit(f"No images found in {kodak_dir}")
    print(f"Kodak eval images: {len(filepaths)}")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    train_dataset = ImageFolder(args.dataset, transform=train_transforms, split="train")
    val_dataset   = ImageFolder(args.dataset, transform=test_transforms,  split="test")
    batch_size    = args.batch_size
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 1. Load anchor model (0.0483 checkpoint)
    model = Ready_Model(0.0483)
    n_linear_before = count_linear(model)

    # 2. Freeze all pretrained weights
    for p in model.parameters():
        p.requires_grad = False

    # 3. Add mask vectors (1 row) to all linear layers — new params, trainable by default
    model = apply_mask_to_all_linear(model, num_bitrates=1)

    # 4. Add adapters to MLP fc1/fc2 — new params, trainable by default
    model = add_adapters_to_mlp(model, alpha=8, r=8, num_bitrates=1)

    n_masked_after = count_masked(model)
    print("Before:", n_linear_before)
    print("After:",  n_masked_after)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    wandb.watch(model, log="all", log_freq=100)

    train_model(
        model,
        device,
        train_loader,
        val_loader,
        args=args,
        filepaths=filepaths,
        baseline_bpp=baseline_bpp,
        baseline_psnr=baseline_psnr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
