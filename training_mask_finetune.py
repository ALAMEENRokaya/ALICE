import math
import random
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import wandb
import sys
from compressai.datasets.utils import ImageFolder
sys.path.append('/home/mmproj01/repos/ALICE')
from utils.Maksadapters import apply_mask_to_all_linear, BaseMaskedFC
from utils.LoadModel import Ready_Model
from eval.evaluate import collect_images, eval_model
import shutil
from pathlib import Path
import json
import matplotlib.pyplot as plt


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
    baseline_bpp = data["bpp"]
    assert len(baseline_psnr) == len(baseline_bpp), "psnr and bpp must have same length"
    return baseline_bpp, baseline_psnr


def wandb_log_psnr_bpp_progress(baseline_bpp, baseline_psnr, epoch_points, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(baseline_bpp, baseline_psnr, marker="o", label="baseline")
    xs = [p["bpp"] for p in epoch_points]
    ys = [p["psnr"] for p in epoch_points]
    ax.scatter(xs, ys, label="trained")
    ax.set_xlabel("BPP")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR vs BPP (epoch {epoch})")
    ax.legend()
    wandb.log({"plots/psnr_vs_bpp_progress": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth.tar"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pth.tar")
        shutil.copyfile(ckpt_path, best_path)


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
        out["loss"] = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out


lambdas = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483]


def train(model, train_loader, optimizer, scheduler, loss_function, device, max_iters=20_000):
    model.train()
    running = {"loss": 0, "mse": 0, "bpp": 0}
    iters = 0
    pbar = tqdm(total=max_iters, desc="Training", leave=False)
    while iters < max_iters:
        for inputs in train_loader:
            bitrate_idx = torch.randint(0, 6, (1,)).item()
            set_bitrate(model, bitrate_idx)
            lmbda = lambdas[bitrate_idx]
            targets = inputs
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_out = loss_function(outputs, targets, lmbda)
            loss_out["loss"].backward()
            optimizer.step()
            scheduler.step()  # step per iteration so T_max=total_iters is respected

            running["loss"] += loss_out["loss"].item()
            running["mse"] += loss_out["mse_loss"].item()
            running["bpp"] += loss_out["bpp_loss"].item()

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
    lmbda = lambdas[bitrate_idx]
    with torch.no_grad():
        for inputs in tqdm(val_loader, desc=f"Validation (idx={bitrate_idx})", leave=False):
            targets = inputs
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss_out = loss_function(outputs, targets, lmbda)
            running["loss"] += loss_out["loss"].item()
            running["mse"] += loss_out["mse_loss"].item()
            running["bpp"] += loss_out["bpp_loss"].item()
            n += 1
    for k in running:
        running[k] /= n
    return running


def train_model(model, device, train_loader, val_loader, args,
                *, filepaths, baseline_bpp, baseline_psnr, batch_size=16, epochs=40,
                max_iters=20_000):
    best_val_loss = float("inf")
    model = model.to(device)
    loss_fn = RateDistortionLoss()

    # All parameters (model weights + masks) are trainable
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # T_max = total iterations across all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * max_iters)

    epoch_points = []
    for epoch in range(epochs):
        train_stats = train(model, train_loader, optimizer, scheduler, loss_fn, device,
                            max_iters=max_iters)
        val_stats = validate(model, val_loader, loss_fn, device, bitrate_idx=5)

        is_best_val = val_stats["loss"] < best_val_loss
        best_val_loss = min(best_val_loss, val_stats["loss"])

        model.eval()
        kodak_by_idx = []
        for idx in range(6):
            set_bitrate(model, idx)
            print("idx", idx, "avg gate on =", gate_sparsity(model))
            metrics = eval_model(model, filepaths)
            metrics["bitrate_idx"] = idx
            metrics["lmbda"] = lambdas[idx]
            kodak_by_idx.append(metrics)

        for m in kodak_by_idx:
            print(
                f"Kodak idx={m['bitrate_idx']} (λ={m['lmbda']}) | "
                f"PSNR: {m['psnr']:.3f}, MS-SSIM: {m['ms-ssim']:.5f}, BPP: {m['bpp']:.4f}"
            )

        track_idx = 5
        kodak_metrics = kodak_by_idx[track_idx]
        epoch_points.append({
            "epoch": epoch + 1,
            "bpp": kodak_metrics["bpp"],
            "psnr": kodak_metrics["psnr"],
        })

        for m in kodak_by_idx:
            wandb.log({
                f"eval/kodak_psnr_idx{m['bitrate_idx']}": m["psnr"],
                f"eval/kodak_bpp_idx{m['bitrate_idx']}": m["bpp"],
                f"eval/kodak_msssim_idx{m['bitrate_idx']}": m["ms-ssim"],
                f"eval/kodak_enc_time_idx{m['bitrate_idx']}": m["encoding_time"],
                f"eval/kodak_dec_time_idx{m['bitrate_idx']}": m["decoding_time"],
            }, step=epoch + 1)

        wandb_log_psnr_bpp_progress(baseline_bpp, baseline_psnr, epoch_points, epoch + 1)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "args": vars(args),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "kodak_by_idx": kodak_by_idx,
                "kodak_metrics": kodak_metrics,
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
            "epoch": epoch + 1,
            "train/loss": train_stats["loss"],
            "train/mse_loss": train_stats["mse"],
            "train/bpp_loss": train_stats["bpp"],
            "val/loss": val_stats["loss"],
            "val/mse_loss": val_stats["mse"],
            "val/bpp_loss": val_stats["bpp"],
            "eval/kodak_psnr": kodak_metrics["psnr"],
            "eval/kodak_msssim": kodak_metrics["ms-ssim"],
            "eval/kodak_bpp": kodak_metrics["bpp"],
            "eval/kodak_enc_time": kodak_metrics["encoding_time"],
            "eval/kodak_dec_time": kodak_metrics["decoding_time"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch + 1)


def count_masked(model):
    return sum(1 for m in model.modules() if isinstance(m, BaseMaskedFC))


def count_linear(model):
    return sum(1 for m in model.modules() if isinstance(m, nn.Linear))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Mask + finetune training (no LoRA).")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("-e", "--epochs", default=13, type=int,
                        help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--val-batch-size", type=int, default=64,
                        help="Validation batch size (default: %(default)s)")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                        help="Patch size for random crop (default: %(default)s)")
    parser.add_argument("--max-iters", type=int, default=20_000,
                        help="Training iterations per epoch (default: %(default)s)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints_mask_finetune",
                        help="Where to save checkpoints")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)

    wandb.init(
        project="image-compression-training",
        name=f"mask_finetune_bs{args.batch_size}",
        config=vars(args)
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(0))

    baseline_json = "/home/mmproj01/repos/ALICE/results/results.json"
    baseline_bpp, baseline_psnr = load_baseline_curve(baseline_json)
    print("Loaded baseline points:", list(zip(baseline_bpp, baseline_psnr)))

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
    val_dataset = ImageFolder(args.dataset, transform=test_transforms, split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                            shuffle=False, num_workers=4)

    # Load pretrained model (weights are trainable by default)
    model = Ready_Model(0.0483)

    n_linear_before = count_linear(model)
    # Apply masks to ALL linear layers — no LoRA, no weight freezing
    model = apply_mask_to_all_linear(model, num_bitrates=6)
    n_masked_after = count_masked(model)
    print(f"Linear layers before: {n_linear_before}")
    print(f"Masked layers after:  {n_masked_after}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
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
        max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
