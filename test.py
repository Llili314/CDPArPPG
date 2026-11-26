import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Load_data_pkl import PKL_COHFACE_train
from data_transform import Normaliztion, ToTensor
import matplotlib.pyplot as plt
from detecta import detect_peaks
from saliencyc_1d import *


from model import ViT_ST_ST_Compact3_TDC_gra_sharp



def get_params(model):
    return sum(p.numel() for p in model.parameters())

def plot_bland_altman(hr_gt_arr, hr_pr_arr, title="Bland–Altman (HR)"):
    hr_gt_arr = np.asarray(hr_gt_arr, dtype=np.float64)
    hr_pr_arr = np.asarray(hr_pr_arr, dtype=np.float64)

    valid = np.isfinite(hr_gt_arr) & np.isfinite(hr_pr_arr)
    hr_gt_arr = hr_gt_arr[valid]
    hr_pr_arr = hr_pr_arr[valid]

    mean_hr = (hr_gt_arr + hr_pr_arr) / 2.0
    diff = (hr_pr_arr - hr_gt_arr)  # Pred - GT

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    plt.figure(figsize=(6, 4))
    plt.scatter(mean_hr, diff, s=12, alpha=0.6)
    plt.axhline(bias, linewidth=1.5)
    plt.axhline(loa_upper, linestyle="--", linewidth=1.5)
    plt.axhline(loa_lower, linestyle="--", linewidth=1.5)

    plt.title(title)
    plt.xlabel("Mean of HR (GT & Pred) [bpm]")
    plt.ylabel("Difference (Pred - GT) [bpm]")

    # 标注
    txt = (
        f"Bias = {bias:.2f}\n"
        f"SD = {sd:.2f}\n"
        f"LoA = [{loa_lower:.2f}, {loa_upper:.2f}]"
    )
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   va="top", ha="left")

    plt.tight_layout()
    plt.show()

def plot_pearson_hist(pearsons, bins=40, title="Pearson Distribution"):
    pearsons = np.asarray(pearsons, dtype=np.float64)
    pearsons = pearsons[np.isfinite(pearsons)]

    plt.figure(figsize=(6, 4))
    plt.hist(pearsons, bins=bins, range=(-1, 1))
    plt.xlim(0.7, 1.2)
    plt.title(title)
    plt.xlabel("Pearson r (per-clip)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_hr_scatter(hr_gt_arr, hr_pr_arr, title="HR Pred vs GT"):
    hr_gt_arr = np.asarray(hr_gt_arr, dtype=np.float64)
    hr_pr_arr = np.asarray(hr_pr_arr, dtype=np.float64)

    valid = np.isfinite(hr_gt_arr) & np.isfinite(hr_pr_arr)
    x = hr_gt_arr[valid]
    y = hr_pr_arr[valid]
    mn = min(np.min(x), np.min(y))
    mx = max(np.max(x), np.max(y))

    plt.figure(figsize=(5, 5))

    plt.scatter(x, y, s=6, alpha=0.2)
    plt.plot([mn, mx], [mn, mx], linewidth=1.5)  # y=x
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.title(title)
    plt.xlabel("GT HR [bpm]")
    plt.ylabel("Pred HR [bpm]")
    plt.tight_layout()
    plt.show()

def plot_all(hr_gt_arr, hr_pr_arr, pearsons, bins=40):
    plot_hr_scatter(hr_gt_arr, hr_pr_arr, title="HR Pred vs GT")
    plot_bland_altman(hr_gt_arr, hr_pr_arr, title="Bland–Altman (HR)")
    plot_pearson_hist(pearsons, bins=bins, title=f"Pearson Distribution (bins={bins})")

@torch.no_grad()
def pearson_per_clip(x, y, eps=1e-8):
    """
    x,y: [B,T]
    return: [B]
    """
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x * x).sum(dim=1) * (y * y).sum(dim=1) + eps)
    return num / den


def hr_from_rppg_fft(sig, fs, f_lo=0.2, f_hi=3):
    """
    FFT 取心率频带(Hz)最大峰值 -> HR(bpm)
    默认频带 0.7~2.5Hz 即 42~150bpm
    """
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    sig = sig * np.hanning(len(sig))

    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spec = np.abs(np.fft.rfft(sig)) ** 2

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return np.nan

    peak = np.argmax(spec[mask])
    peak_freq = freqs[mask][peak]
    return float(peak_freq * 60.0)

def hr_pred(dt):
    peak_values = detect_peaks(dt, mph=0, mpd=10, show=False)
    return len(peak_values)

@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()

    pearsons = []
    neg_pearsons = []
    hr_gt_list = []
    hr_pre_list = []

    for batch in loader:
        x = batch["video_x"].to(device)                 # [B,3,T,H,W]
        ecg = batch["ecg"].to(device)                   # [B,T]
        hr_gt = batch["clip_average_HR"].view(-1).cpu().numpy()
        fs = batch["frame_rate"].view(-1).cpu().numpy()

        rppg, _, features_last1, _, _ = model(x, 2.0)                      # [B,T]

        # per-clip z-norm
        rppg = (rppg - rppg.mean(dim=1, keepdim=True)) / (rppg.std(dim=1, keepdim=True) + 1e-8)
        ecg_n = (ecg - ecg.mean(dim=1, keepdim=True)) / (ecg.std(dim=1, keepdim=True) + 1e-8)

        p = pearson_per_clip(rppg, ecg_n)               # [B]
        pearsons.append(p.cpu().numpy())
        neg_pearsons.append((1.0 - p).cpu().numpy())

        rppg_np = rppg.cpu().numpy()
        ecg_np = ecg_n.cpu().numpy()
        features_last1_np = features_last1.cpu().numpy()
        for i in range(rppg_np.shape[0]):
            peaks = hr_pred(rppg_np[i])
            hr_pre = np.round((fs[i] * 60 * peaks) / len(rppg_np[i]))
            hr_pre_list.append(hr_pre)
            hr_gt_list.append(float(hr_gt[i]))






    #把一个“按 batch 收集的 list”拼成一个完整的一维 NumPy 数组，并且避免空 list 报错
    pearsons = np.concatenate(pearsons) if pearsons else np.array([])
    neg_pearsons = np.concatenate(neg_pearsons) if neg_pearsons else np.array([])

    hr_gt_arr = np.asarray(hr_gt_list, dtype=np.float64)
    hr_pr_arr = np.asarray(hr_pre_list, dtype=np.float64)

    valid = np.isfinite(hr_gt_arr) & np.isfinite(hr_pr_arr)
    hr_gt_arr = hr_gt_arr[valid]
    hr_pr_arr = hr_pr_arr[valid]

    out = {}

    if pearsons.size > 0:
        out["rPPG_Pearson_mean"] = float(np.mean(pearsons))
        out["rPPG_Pearson_std"] = float(np.std(pearsons))
        out["rPPG_NegPearson_mean"] = float(np.mean(neg_pearsons))

    if hr_gt_arr.size > 0:
        diff = hr_pr_arr - hr_gt_arr
        out["HR_MAE"] = float(np.mean(np.abs(diff)))
        out["HR_RMSE"] = float(np.sqrt(np.mean(diff ** 2)))
        out["HR_STD"] = float(np.std(diff))
        out["HR_MAPE(%)"] = float(np.mean(np.abs(diff) / (hr_gt_arr + 1e-8)) * 100.0)  #相对误差，误差占 GT 的比例
        out["HR_Acc@5bpm(%)"] = float(np.mean(np.abs(diff) <= 5.0) * 100.0)    #多少比例的样本，误差 ≤ 5 bpm
        out["HR_Acc@10bpm(%)"] = float(np.mean(np.abs(diff) <= 10.0) * 100.0)  #多少比例的样本，误差 ≤ 10 bpm
        if hr_gt_arr.size >= 2 and np.std(hr_gt_arr) > 1e-6 and np.std(hr_pr_arr) > 1e-6:
            out["HR_Pearson"] = float(np.corrcoef(hr_gt_arr, hr_pr_arr)[0, 1])

        #plot_bland_altman(hr_gt_arr, hr_pr_arr, title="Bland–Altman (HR)")
        #plot_pearson_hist(pearsons, bins=40)
        #plot_hr_scatter(hr_gt_arr, hr_pr_arr)
        plot_all(hr_gt_arr, hr_pr_arr, pearsons, bins=40)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=r'your trained model directory', help="path to .pth (state_dict)")
    parser.add_argument("--pkl_video_dir", type=str, default=r'your test data videl directory', help='RGB video pkl directory')
    parser.add_argument("--pkl_ecg_dir", type=str, default=r'your test data pulse directory',help='Pulse/rPPG pkl directory')
    parser.add_argument("--pkl_hr_dir", type=str, default=r'your test data ground truth directory',help='GT HR pkl directory')
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(128, 128, 128),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=8,
        dropout_rate=0.1,
        theta=0.7
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    ds = PKL_COHFACE_train(
        video_dir=args.pkl_video_dir,
        ecg_dir=args.pkl_ecg_dir,
        hr_dir=args.pkl_hr_dir,
        fps_value=args.fps,
        transform = transforms.Compose([
            Normaliztion(),
            ToTensor()
    ])
    )

    loader = DataLoader(
        ds,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    metrics = evaluate(model, loader, device=device)

    print("===== Evaluation Results =====")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Samples   : {len(ds)}")
    print(f"Params    : {get_params(model) / 1e6:.3f} M")
    for k, v in metrics.items():
        print(f"{k:>18}: {v:.6f}")


if __name__ == "__main__":
    main()
