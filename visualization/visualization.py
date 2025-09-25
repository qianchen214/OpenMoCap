#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, warnings, argparse
import numpy as np

def _import_matplotlib():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    return plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection

try:
    from tqdm import tqdm
except Exception:
    class tqdm:
        def __init__(self, total=None, desc=None): self.n=0
        def update(self, n=1): self.n += n
        def close(self): pass

EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (7, 10), (5, 8), (8, 11),
    (6, 9), (9, 12), (12, 15), (9, 13), (9, 14), (13, 16), (16, 18), (18, 20), (20, 22),
    (14, 17), (17, 19), (19, 21), (21, 23)
]
JOINT_SIZE = 6
BONE_WIDTH = 1.2

BONE_COLOR  = '#66B2FF'
JOINT_COLOR = '#0B5FFF'

UP_AXIS           = 'z'
DRAW_GROUND       = True
GROUND_FACE       = '#F2F4F7'
GROUND_EDGE       = '#E5E7EB'
GROUND_FACE_ALPHA = 0.08
GROUND_GRID       = True
GROUND_GRID_N     = 10
GROUND_PUSH_DOWN  = 0.03
Z_PERSON = 1000
Z_GROUND = -1000

def to_TJ3(arr):
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape == (3,):
        a = a[None, None, :]
    if a.ndim == 2 and a.shape[1] == 3:
        a = a[None, ...]
    if a.ndim == 3:
        if a.shape[-1] == 3:
            return a
        if a.shape[0] == 3:
            return np.transpose(a, (2,1,0))
        if a.shape[1] == 3:
            return np.transpose(a, (2,0,1))
    raise ValueError(f"Unsupported joints shape: {a.shape}")

def _add_ground_plane(ax, joints, Poly3DCollection):
    mins = joints.min(axis=(0,1)); maxs = joints.max(axis=(0,1))
    span = float((maxs - mins).max()); pad = 0.05 * span; push = GROUND_PUSH_DOWN * span
    if UP_AXIS.lower() == 'y':
        x_min, x_max = joints[...,0].min(), joints[...,0].max()
        z_min, z_max = joints[...,2].min(), joints[...,2].max()
        y_min = joints[...,1].min(); y_g = y_min - push
        x_min -= pad; x_max += pad; z_min -= pad; z_max += pad
        verts = [[(x_min, y_g, z_min),(x_max, y_g, z_min),(x_max, y_g, z_max),(x_min, y_g, z_max)]]
        poly = Poly3DCollection(verts, facecolors=GROUND_FACE, edgecolors=GROUND_EDGE,
                                linewidths=1.0, alpha=GROUND_FACE_ALPHA, zorder=Z_GROUND)
        ax.add_collection3d(poly)
        if GROUND_GRID:
            nx = nz = int(GROUND_GRID_N)
            for i in range(nx + 1):
                x = x_min + (x_max - x_min) * i / nx
                ax.plot([x, x], [y_g, y_g], [z_min, z_max], color=GROUND_EDGE, linewidth=0.6, alpha=0.6, zorder=Z_GROUND)
            for j in range(nz + 1):
                z = z_min + (z_max - z_min) * j / nz
                ax.plot([x_min, x_max], [y_g, y_g], [z, z], color=GROUND_EDGE, linewidth=0.6, alpha=0.6, zorder=Z_GROUND)
    else:
        x_min, x_max = joints[...,0].min(), joints[...,0].max()
        y_min, y_max = joints[...,1].min(), joints[...,1].max()
        z_min = joints[...,2].min(); z_g = z_min - push
        x_min -= pad; x_max += pad; y_min -= pad; y_max += pad
        verts = [[(x_min, y_min, z_g),(x_max, y_min, z_g),(x_max, y_max, z_g),(x_min, y_max, z_g)]]
        poly = Poly3DCollection(verts, facecolors=GROUND_FACE, edgecolors=GROUND_EDGE,
                                linewidths=1.0, alpha=GROUND_FACE_ALPHA, zorder=Z_GROUND)
        ax.add_collection3d(poly)
        if GROUND_GRID:
            nx = ny = int(GROUND_GRID_N)
            for i in range(nx + 1):
                x = x_min + (x_max - x_min) * i / nx
                ax.plot([x, x], [y_min, y_max], [z_g, z_g], color=GROUND_EDGE, linewidth=0.6, alpha=0.6, zorder=Z_GROUND)
            for j in range(ny + 1):
                y = y_min + (y_max - y_min) * j / ny
                ax.plot([x_min, x_max], [y, y], [z_g, z_g], color=GROUND_EDGE, linewidth=0.6, alpha=0.6, zorder=Z_GROUND)

def _set_axes_equal(ax, xyz):
    mins = xyz.min(axis=(0,1)); maxs = xyz.max(axis=(0,1))
    center = (mins + maxs) / 2.0; r = (maxs - mins).max()
    if r <= 0: r = 1.0
    r *= 0.55
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)

def _build_scene(joints, args, plt, Poly3DCollection):
    fig = plt.figure(figsize=tuple(args.figsize), dpi=args.dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')

    ax.set_axis_off(); ax.grid(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        except Exception:
            pass

    if DRAW_GROUND:
        _add_ground_plane(ax, joints, Poly3DCollection)

    scat = ax.scatter([], [], [], s=JOINT_SIZE, c=JOINT_COLOR, linewidths=0,
                      zorder=Z_PERSON, depthshade=False)
    lines = [ax.plot([], [], [], lw=BONE_WIDTH, c=BONE_COLOR, solid_capstyle='round',
                     zorder=Z_PERSON-1)[0] for _ in EDGES]
    for ln in lines:
        try: ln.set_sort_zpos(-1e6)
        except Exception: pass
    try: scat.set_sort_zpos(1e6)
    except Exception: pass

    _set_axes_equal(ax, joints)
    return fig, ax, scat, lines

def _make_update(joints, scat, lines):
    T, J, _ = joints.shape
    def update(t):
        P = joints[t]
        scat._offsets3d = (P[:,0], P[:,1], P[:,2])
        for k,(i,j) in enumerate(EDGES):
            lines[k].set_data([P[i,0], P[j,0]], [P[i,1], P[j,1]])
            lines[k].set_3d_properties([P[i,2], P[j,2]])
        return [scat] + lines
    return update

def render_npz_to_video(npz_path, key="pred_j", out_path=None,
                        fps=30, skip=1, dpi=180, figsize=(8,8),
                        elev=20.0, azim=-60.0):
    """
    Render NPZ[key] to MP4 using libx264
    """
    plt, FuncAnimation, FFMpegWriter, PillowWriter, Poly3DCollection = _import_matplotlib()

    try:
        from tqdm import tqdm
        def make_pbar(total, desc): return tqdm(total=total, desc=desc, unit="f", dynamic_ncols=True)
    except Exception:
        class _Dummy:
            def __init__(self, total=None, desc=None): pass
            def update(self, n=1): pass
            def close(self): pass
        def make_pbar(total, desc): return _Dummy()

    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key not found in {npz_path}: {key}. Available keys: {list(data.keys())}")
    joints = to_TJ3(data[key])
    T, _, _ = joints.shape

    out_root = os.path.splitext(out_path if out_path else npz_path)[0]

    class _Args: pass
    args = _Args()
    args.fps = fps
    args.skip = max(1, int(skip))
    args.dpi = dpi
    args.figsize = figsize
    args.elev = elev
    args.azim = azim
    args.axis = False
    args.out = out_root + ".mp4"

    fig, ax, scat, lines = _build_scene(joints, args, plt, Poly3DCollection)
    update_fn = _make_update(joints, scat, lines)

    frame_indices = list(range(0, T, args.skip))

    # libx264 + CRF
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23"]
    )
    pbar = make_pbar(len(frame_indices), "Rendering[MP4]")
    try:
        with writer.saving(fig, args.out, args.dpi):
            for t in frame_indices:
                update_fn(t)
                writer.grab_frame()
                pbar.update(1)
    finally:
        pbar.close()
        plt.close(fig)

    print("✅ Saved:", args.out)
    return args.out

def _build_cli():
    ap = argparse.ArgumentParser(description="Render a .npz key to MP4.")
    ap.add_argument('--npz', required=True, help="Path to the input .npz file")
    ap.add_argument('--key', required=True, help="Key to read from the predicted NPZ when rendering, such as pred_j / J / joints")
    ap.add_argument('--out', default=None, help="Output video path (defaults to an .mp4 with the same name as the NPZ)")
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--skip', type=int, default=4)
    ap.add_argument('--dpi', type=int, default=180)
    ap.add_argument('--figsize', type=float, nargs=2, default=(8,8))
    ap.add_argument('--elev', type=float, default=20.0)
    ap.add_argument('--azim', type=float, default=-60.0)
    return ap

def main():
    args = _build_cli().parse_args()
    render_npz_to_video(
        npz_path=args.npz,
        key=args.key,
        out_path=args.out,
        fps=args.fps, skip=args.skip,
        dpi=args.dpi, figsize=tuple(args.figsize),
        elev=args.elev, azim=args.azim
    )

if __name__ == "__main__":
    main()
