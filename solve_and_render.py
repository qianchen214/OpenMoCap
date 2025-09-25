#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import numpy as np
import time, random
import torch
from torch.utils.data import DataLoader

from dataset.rotate_dataset import RotateDataset_perfile
from util.utils import get_transfer_matrices_gpu
from models import model_position
from models.model_6d import AdapterNet

from visualization.visualization import render_npz_to_video

def _sync_cuda(device: torch.device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()

def prepare_model(chkpt_dir, arch="mae_vit_base_patch16", device="cpu"):
    model = getattr(model_position, arch)()
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint, strict=True)
    print("[Position] Model loaded:", msg)
    return model.to(device).eval()

def prepare_angle_model(chkpt_angle, device="cpu"):
    model_ang = AdapterNet()
    checkpoint_angle = torch.load(chkpt_angle, map_location="cpu")
    model_ang.load_state_dict(checkpoint_angle, strict=True)
    print("[Rotation] Model loaded.")
    return model_ang.to(device).eval()

@torch.no_grad()
def run_on_file(npz_file, out_dir, model, model_ang, device,
                t_pos_marker, ref_idx, batch_size=512):
    dataset = RotateDataset_perfile(npz_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                            pin_memory=True, drop_last=False)

    _sync_cuda(device)
    t0 = time.perf_counter()

    accum_iter = 0
    rot_loss = 0.0
    pos_loss = 0.0

    # all_preds = []
    all_input_m = []
    all_pred_m = []
    all_pred_j_p = []
    all_pred_j_r = []

    for samples in dataloader:
        cur_data = {
            "M":  samples["M"].to(device, non_blocking=True),
            "M1": samples["M1"].to(device, non_blocking=True),
            "J":  samples["J"].to(device, non_blocking=True),
        }
        input_m = samples["M1"].to(device, non_blocking=True)
        all_input_m.append(input_m.detach().cpu().numpy())
        J_R = samples["JR_rot_mat"].to(device, non_blocking=True)

        _, _, joint_ploss, _, pred, _ = model(cur_data)
        pred_j_p = pred[:, 56:].detach().cpu().numpy()
        pred_m = pred[:, :56].detach().cpu().numpy()
        # all_preds.append(pred_j)
        all_pred_j_p.append(pred_j_p)
        all_pred_m.append(pred_m)

        R, T = get_transfer_matrices_gpu(
            pred[:, ref_idx], t_pos_marker.unsqueeze(0).repeat(pred.shape[0], 1, 1)
        )
        new_pos = torch.matmul(pred, R.transpose(1, 2)) + T[:, None, :]
        new_rot_mat = torch.einsum("nij,nkjl->nkil", R, J_R)
        pred_j_r, loss = model_ang(new_pos, new_rot_mat)
        all_pred_j_r.append(pred_j_r.detach().cpu().numpy())

        accum_iter += 1
        rot_loss += loss.item()
        pos_loss += joint_ploss.item()

    # all_preds = np.concatenate(all_preds, axis=0)
    all_input_m = np.concatenate(all_input_m, axis=0)
    all_pred_m = np.concatenate(all_pred_m, axis=0)
    all_pred_j_p = np.concatenate(all_pred_j_p, axis=0)
    all_pred_j_r = np.concatenate(all_pred_j_r, axis=0)

    _sync_cuda(device)
    num_frames = int(all_pred_m.shape[0])
    solve_secs = time.perf_counter() - t0

    basename = os.path.splitext(os.path.basename(npz_file))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}.npz")
    # np.savez(out_path, pred_j=all_preds)
    np.savez(out_path, input_m=all_input_m, pred_m=all_pred_m, pred_j_p=all_pred_j_p, pred_j_r=all_pred_j_r)

    print(f"[Solve] {basename}: {solve_secs:.3f}s | output={out_path} | frames={num_frames}")
    print(f'joint pos error: {pos_loss / accum_iter}, joint angle error: {rot_loss / accum_iter}')

    return out_path

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Run solver on .npz (single file or directory). Optionally render to MP4 with --render."
    )
    # solve
    ap.add_argument("--data_dir", default="./test_cases/sfu",
                    help="Directory containing input .npz files; mutually exclusive with --file (if both are provided, --file takes precedence).")
    ap.add_argument("--file", help="Process a single .npz file only")
    ap.add_argument("--out_dir", default="./results", help="Output directory for predicted .npz files")
    ap.add_argument("--model_pos", default="./openmocap_position/model_pos.pth", help="Path to the position model checkpoint")
    ap.add_argument("--model_rot", default="./marker_joint_6d/model_ang.pth", help="Path to the rotation model checkpoint")
    ap.add_argument("--arch", default="mae_vit_base_patch16")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--random_one", action="store_true",
                    help="If set, randomly select only one .npz file from the dataset directory.")


    # render
    ap.add_argument("--render", action="store_true", help="Enable rendering")
    ap.add_argument("--key", default="pred_j_p", help="Key to read from the predicted NPZ when rendering")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--figsize", type=float, nargs=2, default=(8,8))
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    return ap

def main():
    args = build_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    model = prepare_model(args.model_pos, arch=args.arch, device=device)
    model_ang = prepare_angle_model(args.model_rot, device=device)

    ref_idx = [54, 3, 46, 32, 36, 21, 7, 11]
    t_pose = np.load(os.path.join("util", "t_pos_marker.npy"))
    t_pos_marker = torch.tensor(t_pose, dtype=torch.float32, device=device)

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        if args.random_one and files:
            files = [random.choice(files)]
    print(f"[Input] Found {len(files)} file(s).")

    for in_npz in files:
        basename = os.path.splitext(os.path.basename(in_npz))[0]
        mp4_path = os.path.join(args.out_dir, f"{basename}.mp4")

        if os.path.exists(mp4_path):
            print(f"[Skip] {mp4_path} already exists, skipping.")
            continue
        
        pred_npz = run_on_file(
            npz_file=in_npz,
            out_dir=args.out_dir,
            model=model,
            model_ang=model_ang,
            device=device,
            t_pos_marker=t_pos_marker,
            ref_idx=ref_idx,
            batch_size=args.batch_size
        )

        if args.render:
            try:
                _ = render_npz_to_video(
                    npz_path=pred_npz,
                    key=args.key,
                    out_path=os.path.splitext(pred_npz)[0] + ".mp4",
                    fps=args.fps, skip=args.skip,
                    dpi=args.dpi, figsize=tuple(args.figsize),
                    elev=args.elev, azim=args.azim
                )
            except Exception as e:
                print(f"[Render Error] {pred_npz}: {e}")

if __name__ == "__main__":
    main()
