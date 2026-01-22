import json
import matplotlib.pyplot as plt


def plot_psnr_bpp(in_json="metrics_results.json", out_png="psnr_curve.png"):
    with open(in_json, "r") as f:
        data = json.load(f)

    lams = data["lambdas"]
    bpps = data["metrics"]["bpp"]
    psnrs = data["metrics"]["psnr"]

    plt.figure()
    plt.plot(bpps, psnrs, marker="o")

    for lam, x, y in zip(lams, bpps, psnrs):
        plt.annotate(f"{lam:g}", (x, y), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("BPP")
    plt.ylabel("PSNR (dB)")
    plt.title(f'{data["model"]}: PSNR vs BPP ({data["dataset"]})')
    plt.grid(True)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_png}")
