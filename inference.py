# -*- coding: utf-8 -*-
"""
Inference script for Sparse2Inverse
Dedicated for noise_intensity=6000
Automatically loads and runs inference for:
- p2p_best_ssim_XXXX.pth
- p2p_best_lpips_XXXX.pth
- s2i_best_ssim_XXXX.pth
- s2i_best_lpips_XXXX.pth
Supports methods:
- P2P
- S2I
- S2I_ds   (double-split)
"""

from operator import gt

import torch, os, re, argparse
import numpy as np
from torch.utils.data import DataLoader

from models import *
from utils import *
from utils_lodopab import get_images_from_pt

import lpips
from piq import haarpsi
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
import re
# ------------------------------------------------------------
# ARGPARSE
# ------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-angles","--angles",type=int,default=32)
parser.add_argument("-l","--loss_variant",type=str,default="MSE_data")
parser.add_argument("-m","--method",type=str,default="P2P")   # P2P | S2I | S2I_ds
parser.add_argument("-lr","--learning_rate",type=float,default=1e-4)
parser.add_argument("-grid_size","--grid_size",type=int,nargs="+",default=[3])
parser.add_argument("-i","--fill_zeros",action="store_true")
parser.add_argument("-r","--random_mask",action="store_true")
parser.add_argument("--correlated_noise",action="store_true")
parser.add_argument("-ni","--noise_intensity",type=float,default=6000.0, help="Noise intensity (fixed to 6000 if not specified)")
parser.add_argument("-gaussian_noise_std","--gaussian_noise_std",type=float,default=.1)
args = parser.parse_args()
args.fill_zeros = True
# fixed noise intensity for this script

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8

# ------------------------------------------------------------
# FIND LATEST WEIGHTS
# ------------------------------------------------------------
def find_latest(weights_dir, tag):
    pat = re.compile(fr"{tag}_(\d+)\.pth")
    print(pat)
    best = (-1, None)
    for f in os.listdir(weights_dir):
        m = pat.match(f)
        if m:
            e = int(m.group(1))
            if e > best[0]:
                best = (e, f)
    if best[1] is None:
        print(f" Missing {tag}")
        return None, None
    return os.path.join(weights_dir, best[1]), best[0]

# ------------------------------------------------------------
# EXPERIMENT NAME
# ------------------------------------------------------------
if args.correlated_noise:
    experiment_name = (
        f"{args.method}_gridsize_{args.grid_size}_loss_one_grad_step_"
        f"{args.loss_variant}_"
        f"lr_{args.learning_rate}_angles_{args.angles}_"
        f"random_mask_{args.random_mask}_interpolate_{args.fill_zeros}_"
        f"noisetype_{args.correlated_noise}_gaussian_std_{args.gaussian_noise_std}"
    )
else:
    experiment_name = (
        f"{args.method}_gridsize_{args.grid_size}_loss_one_grad_step_"
        f"{args.loss_variant}_"
        f"lr_{args.learning_rate}_angles_{args.angles}_"
        f"random_mask_{args.random_mask}_interpolate_{args.fill_zeros}_{args.noise_intensity}"
    )

print("Experiment name:", experiment_name)

# ------------------------------------------------------------
# AUTO DISCOVER WEIGHT FOLDER BASED ON ARGUMENTS
# ------------------------------------------------------------
def find_matching_weight_dir(base_dirs, args):
    """
    Scan weight folders and pick the one that matches:
    method, grid(s), angles, mask, interpolate, and optionally intensity.
    Handles both legacy naming (gridsize_2) and modern (gridsize_[2]).
    """

    method = args.method
    print('method ist: ', method)
    grids_to_match = [str(g) for g in args.grid_size]  # ["2", "3", ...]
    angles_str = f"angles_{args.angles}"
    mask_str = f"random_mask_{args.random_mask}"
    interp_str = f"interpolate_{args.fill_zeros}"
    if args.correlated_noise==False:
        intensity_str = f"intensity_{args.noise_intensity:.1f}"

    print("\n Searching matching weight folder...\n")

    for base in base_dirs:
        for folder in os.listdir(base):
            print('folder is ', folder)
            full = os.path.join(base, folder)
            print(full)
            if not os.path.isdir(full):
                continue

            # Check method, angles, mask
            conditions = [
                folder.startswith(method + "_grid"),
                angles_str in folder,
                mask_str in folder,
            ]

            # Check grids: either _2_, _3_, or [_2, 3], etc.
            grid_match = any(f"gridsize_{g}" in folder or f"gridsize_[{g}]" in folder for g in grids_to_match)
            conditions.append(grid_match)

            # interpolate may be missing in older folders
            if "interpolate" in folder:
                conditions.append(interp_str in folder)
            else:
                conditions.append(True)

            # intensity may be missing in older folders
            if args.correlated_noise==False:
                if "intensity" in folder:
                    conditions.append(intensity_str in folder)
                else:
                    conditions.append(True)

            if all(conditions):
                print(f"Matched weight folder:\n{full}\n")
                return full

    raise FileNotFoundError(
        f"No matching weight folder found for:\n"
        f"method={method}, grids={args.grid_size}, angles={args.angles}, "
        f"mask={args.random_mask}, interpolate={args.fill_zeros}, "
        f"intensity={args.noise_intensity}"
    )

# --------------------------------------------------------------------
# Select correct base directory automatically by angle and noise type   
# --------------------------------------------------------------------
bases = []
if args.angles == 16 and args.correlated_noise == True:
    bases = [f"../outputs/weights_paper_16_correlated_noise_{args.gaussian_noise_std}"]
elif args.angles == 16 and args.correlated_noise == False:
    bases = [f"../outputs/weights_paper_16_uncorrelated_noise_{args.noise_intensity}"]
elif args.angles == 32 and args.correlated_noise == True:
    bases = [f"../outputs/weights_paper_32_correlated_noise_{args.gaussian_noise_std}"]
elif args.angles == 32 and args.correlated_noise == False:
    bases = [f"../outputs/weights_paper_32_uncorrelated_noise_{args.noise_intensity}"]
else:
    raise ValueError("Only angles 16 or 32 supported for these weights.")

final_weights_dir = find_matching_weight_dir(bases, args)
print("Using weight directory:", final_weights_dir)

# ------------------------------------------------------------
# Output root
# ------------------------------------------------------------
if args.correlated_noise == False:
    csv_path = f"./predictions_paper_uncorrelated_noise_{args.angles}_{args.noise_intensity}/inferences_summary.csv"
    out_root = os.path.join(f"./predictions_paper_uncorrelated_noise_{args.angles}_{args.noise_intensity}", experiment_name)
else:
    csv_path = f"./predictions_paper_correlated_noise_{args.correlated_noise}_{args.angles}/inferences_summary.csv"
    out_root = os.path.join(f"./predictions_paper_correlated_noise_{args.correlated_noise}_{args.angles}", experiment_name)

print('output ', out_root)
os.makedirs(out_root, exist_ok=True)

# ------------------------------------------------------------
# Model weights inside the selected folder
# ------------------------------------------------------------
print("Final weights directory:", os.listdir(final_weights_dir))
models = {
    "p2p_ssim": find_latest(final_weights_dir, "p2p_best_ssim"),
    "s2i_ssim": find_latest(final_weights_dir, "s2i_best_ssim"),
    "ii_ssim": find_latest(final_weights_dir, "ii_best_ssim"),
    }
open(os.path.join(out_root, "experiment_log.txt"), "a").write(f"{experiment_name} | args={vars(args)} | weights={models} | weights_dir={final_weights_dir}\n")
# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print("Loading test data...")
images = get_images_from_pt("/home/nadja/Documents/Projects/gt_pt", amount_of_images="all", scale_number=1)
images = rescale_images(images, device, target_size=(np.shape(images)[0], 336, 336))
images_test = images[:400]

sinograms_test = torch.tensor(
    create_noisy_sinograms_poisson(
        images_test, args.angles,
        photon_count=args.noise_intensity,
        correlated_noise=args.correlated_noise,
        gaussian_noise_std=args.gaussian_noise_std
    )
)

Data_loader_test = DataLoader(
    torch.utils.data.TensorDataset(sinograms_test, images_test),
    batch_size=batch_size, shuffle=False
)

# ------------------------------------------------------------
# RUN ALL INFERENCES
# ------------------------------------------------------------
def compute_metrics(recos, Ims, device, lpips_fn):
    ssim_list, psnr_list, mse_list, lpips_list, haarpsi_list = [], [], [], [], []
    all_imgs = []

    for i in range(recos.shape[0]):
        reco = recos[i, 0].cpu()
        gt = Ims[i].cpu()
        all_imgs.append(reco.unsqueeze(0))

        reco_np = reco.numpy()
        reco_np = np.clip(reco_np, 0.0, 1.0)
        gt_np = gt.numpy()
        gt_np = np.clip(gt_np, 0.0, 1.0)

        ssim_list.append(structural_similarity(gt_np, reco_np, data_range=gt_np.max()-gt_np.min()))
        psnr_list.append(peak_signal_noise_ratio(gt_np, reco_np))
        mse_list.append(np.mean((gt_np - reco_np)**2))

        reco_lp = (reco.unsqueeze(0).unsqueeze(0)*2 - 1).to(device)
        gt_lp = (gt.unsqueeze(0).unsqueeze(0)*2 - 1).to(device)
        lpips_list.append(lpips_fn(reco_lp, gt_lp).item())

        reco_haar = reco.unsqueeze(0).unsqueeze(0).to(device)
        gt_haar = gt.unsqueeze(0).unsqueeze(0).to(device)
        reco_haar = reco_haar.clamp(0.0, 1.0)
        gt_haar = gt_haar.clamp(0.0, 1.0)
        haarpsi_list.append(haarpsi(reco_haar, gt_haar, data_range=1.0).item())

    return torch.cat(all_imgs, 0), np.array(ssim_list), np.array(psnr_list), np.array(mse_list), np.array(lpips_list), np.array(haarpsi_list)

lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_fn.eval()

for name, (wfile, epoch) in models.items():
    if wfile is None:
        continue

    print(f"\n▶ Running {name} (epoch {epoch})")
    save_dir = os.path.join(out_root, name)
    os.makedirs(save_dir, exist_ok=True)

    # build model
    if args.method == "P2P":
        N2I = Proj2Proj(random=args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
    elif args.method == "S2I":
        N2I = Sparse2Inverse_p2p(random=args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
    elif args.method == "S2I_ds":
        N2I = Sparse2Inverse_ds_all_combinations(random = args.random_mask, grid_size=args.grid_size, fill_zeros=args.fill_zeros)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    N2I.net_denoising.load_state_dict(torch.load(wfile, map_location=device))
    N2I.net_denoising.eval()

    # inference types direct vs average vs P-invariant
    with torch.no_grad():
        if name.startswith("p2p"):
            print(name)
            full_recos, _, Ims, _ = validate_direct(Data_loader_test, N2I)
        elif name.startswith("s2i"):
            print(name)
            if args.method == "S2I_ds":
                full_recos, _, Ims, _ = validate_average_ds(Data_loader_test, N2I)
            else:
                full_recos, _, Ims, _ = validate_average(Data_loader_test, N2I)
        elif name.startswith("ii"):
            print(name)
            if args.method == "S2I_ds":
                full_recos, _, Ims, _ = validate_P_invariant_doublesplit(Data_loader_test, N2I)
            else:
                full_recos, _, Ims, _ = validate_P_invariant(Data_loader_test, N2I)


    all_imgs, ssim, psnr, mse, lpips_vals, haarpsi_vals = compute_metrics(full_recos, Ims, device, lpips_fn)

    torch.save(all_imgs, os.path.join(save_dir, "all_reconstructions.pt"))
    np.savez(os.path.join(save_dir, "metrics.npz"), ssim=ssim, psnr=psnr, mse=mse, lpips=lpips_vals, haarpsi=haarpsi_vals)

    print(f"{name} finished. Mean SSIM: {np.mean(ssim):.4f}")

    # write summary CSV
    summary_csv = csv_path
    df_row = pd.DataFrame([{
        "method_name": name,
        "experiment_name": experiment_name,
        "grid_size": "_".join(map(str, args.grid_size)),
        "angles": args.angles,
        "noise_intensity": args.noise_intensity,
        "ssim": float(np.mean(ssim)),
        "psnr": float(np.mean(psnr)),
        "lpips": float(np.mean(lpips_vals)),
        "haarpsi": float(np.mean(haarpsi_vals))
    }])
    if os.path.exists(summary_csv):
        df_row.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(summary_csv, mode="w", header=True, index=False)

    print(f" Metrics appended to {summary_csv}")

