"""Simple CPU runner that bypasses hydra to avoid signal handler conflicts."""
import os, sys, signal

# Install spconv shim before any Pointcept imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import install_spconv_shim
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Pointcept'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt'))

import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from omegaconf import OmegaConf

# Ignore SIGSEGV from background threads (pycolmap native lib cleanup)
# This is safe because the segfault occurs in a destructor thread, not main logic
signal.signal(signal.SIGSEGV, signal.SIG_IGN)

sys.path.append('vggt')

from feedforward import FeedForward_Model, preprocess
from matching import init_match_models
from sfm.sfm_func import run_sfm
from utils.basic import set_seed, Print
from utils.io import save_xyzrgb_to_ply
from tqdm import tqdm

INTMAX = 1e6

def filter_points(points, confs=None, masks=None, max_pts_num=None, conf_quantile_thresh=0.2):
    if masks is None:
        masks = torch.ones_like(confs, dtype=torch.bool)
    if confs is not None:
        if confs[masks].numel() > INTMAX:
            stride = int(confs[masks].numel() / INTMAX) * 5
            conf_thresh = torch.quantile(confs[masks][::stride], conf_quantile_thresh)
        else:
            conf_thresh = torch.quantile(confs[masks], conf_quantile_thresh)
        masks = masks & (confs >= conf_thresh)
    if max_pts_num is not None:
        if masks.sum() > max_pts_num:
            masks_ = masks.view(-1)
            indices = torch.nonzero(masks_, as_tuple=False)
            indices = indices[torch.randperm(indices.shape[0])[:max_pts_num]]
            masks_ = torch.zeros_like(masks_, dtype=torch.bool)
            masks_[indices] = True
            masks = masks_.view_as(masks)
    return masks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='examples/scannetpp-9bb22982672c69bf')
    parser.add_argument('--output_dir', type=str, default='outputs/demo_cpu')
    parser.add_argument('--max_width', type=int, default=640)
    parser.add_argument('--max_images', type=int, default=8)
    parser.add_argument('--matcher', type=str, default='romav2-base')
    parser.add_argument('--ggpt_refine', action='store_true', default=False)
    parser.add_argument('--max_pts_num', type=int, default=1000000)
    parser.add_argument('--conf_quantile_thresh', type=float, default=0.2)
    args = parser.parse_args()

    # Load config from demo.yaml
    cfg = OmegaConf.load('configs/demo.yaml')
    cfg.image_dir = args.image_dir
    cfg.match_config.max_width = args.max_width
    cfg.match_config.models = [args.matcher]
    cfg.common_config.ggpt_refine = args.ggpt_refine
    cfg.common_config.max_pts_num = args.max_pts_num
    cfg.common_config.conf_quantile_thresh = args.conf_quantile_thresh
    cfg.common_config.reduce_memory = True

    set_seed(cfg.common_config.seed)
    device = 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load FeedForward model
    import cv2
    Print(f"Initializing FeedForward model: {cfg.feedforward_config.model}")
    ff_model = FeedForward_Model(cfg.feedforward_config).to(device)
    ff_model.eval()

    # Load images
    images = []
    for ii, image_path in enumerate(sorted(glob(os.path.join(args.image_dir, "*")))):
        if image_path.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(image_path)[:, :, ::-1]
            if ii == 0:
                h, w = img.shape[:2]
            elif h != img.shape[0] or w != img.shape[1]:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            images.append(img)
            if len(images) >= args.max_images:
                break
    Print(f"Loaded {len(images)} images, original size: {h}x{w}")

    images = torch.from_numpy(np.stack(images, axis=0)).float() / 255.0
    H, W = images.shape[1], images.shape[2]
    if W > args.max_width:
        H = int(round(H * args.max_width / W))
        W = args.max_width
    newH, newW = int(round(H / 14) * 14), int(round(W / 14) * 14)
    images = F.interpolate(images.permute(0, 3, 1, 2), size=(newH, newW), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    Print(f"Resized images to: {newH}x{newW}")

    images = images.to(device)
    with torch.no_grad():
        ff_outputs = ff_model(images, preprocessed=False)
    Print("FeedForward inference done.")

    del ff_model
    Print("Deleted ff_model to save memory.")

    # Dense matching + SfM
    match_models = init_match_models(cfg.match_config.models, device=device)
    Print(f"Initialized Matching models: {list(match_models.keys())}")

    sfm_outputs = run_sfm(images, ff_outputs, match_models, cfg)

    if not sfm_outputs['points_success']:
        Print("SfM fails. The input is a hard case for geometry reconstruction :)")
        return

    del match_models
    Print("SfM done.")

    # Save outputs
    sfm_masks = filter_points(sfm_outputs['points'], None, sfm_outputs['point_masks'],
                              args.max_pts_num, args.conf_quantile_thresh)
    save_xyzrgb_to_ply(
        points=sfm_outputs['points'][sfm_masks],
        rgb=ff_outputs['images_ff'][sfm_masks],
        filename=os.path.join(args.output_dir, 'sfm_dlt_points.ply'))
    Print(f"Saved SfM DLT points to {os.path.join(args.output_dir, 'sfm_dlt_points.ply')}")

    ff_masks = filter_points(ff_outputs['points'], ff_outputs['points_conf'], None,
                             args.max_pts_num, args.conf_quantile_thresh)
    save_xyzrgb_to_ply(
        points=ff_outputs['points'][ff_masks],
        rgb=ff_outputs['images_ff'][ff_masks],
        filename=os.path.join(args.output_dir, 'ff_points.ply'))
    Print(f"Saved feedforward points to {os.path.join(args.output_dir, 'ff_points.ply')}")

    if args.ggpt_refine:
        Print("Loading GGPT model for refinement...")
        from ggpt.model.base import BasePredictor
        from ggpt.dataloader.demo_dataset import DemoDataset
        from utils.points import aggregate_chunks
        from sfm.run_benchmark_sfm import move_to_device
        import time

        ptv3_config = OmegaConf.to_container(cfg.ggptmodel_config.ptv3_config, resolve=True)
        ptv3_config['enable_flash'] = False  # No flash attention on CPU

        ggpt_model = BasePredictor(
            grid_resolution=cfg.ggptmodel_config.grid_resolution,
            type_embed_dim=cfg.ggptmodel_config.type_embed_dim,
            sinusoidal_dim=cfg.ggptmodel_config.sinusoidal_dim,
            zero_init=cfg.ggptmodel_config.zero_init,
            predict_residual=cfg.ggptmodel_config.predict_residual,
            backbone_type=cfg.ggptmodel_config.backbone_type,
            ptv3_config=ptv3_config,
            head_width=cfg.ggptmodel_config.head_width,
        ).eval()

        ckpt = torch.load(cfg.common_config.ggpt_ckpt, map_location='cpu')
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        ggpt_model.load_state_dict(ckpt, strict=True)
        Print(f"Loaded GGPT model from {cfg.common_config.ggpt_ckpt}")

        demo_dataset = DemoDataset(name='demo', ff_data=ff_outputs, geo_data=sfm_outputs)
        scene_chunks, scene = demo_dataset[0]
        chunks_batch = [[chunk] for chunk in scene_chunks]
        to_collect = {'ff_pts': [], 'ff_pts_conf': []}
        t0 = time.time()
        for chunk_batch in tqdm(chunks_batch, desc="GGPT inference"):
            chunk_batch = move_to_device(chunk_batch, device)
            with torch.no_grad():
                out = ggpt_model(chunk_batch)
            to_collect['ff_pts'].append(demo_dataset.unnormalize_pts(chunk_batch[0], out['ff_pts_out']))
            to_collect['ff_pts_conf'].append(out['ff_pts_conf_out'])

        ff_pts_all = torch.cat(to_collect['ff_pts'], dim=0)
        ff_pts_conf_all = torch.cat(to_collect['ff_pts_conf'], dim=0)
        msks_in_scene = torch.stack([chunk['msks_in_scene'] for chunk in scene_chunks], dim=0).to(device)
        pred_pts, pred_confs, pred_mask = aggregate_chunks(ff_pts_all, ff_pts_conf_all, msks_in_scene, scene)
        t1 = time.time()
        Print(f"GGPT inference done in {t1 - t0:.2f}s.")

        # Save GGPT refined points
        pred_mask = filter_points(pred_pts, pred_confs, None, args.max_pts_num, args.conf_quantile_thresh)
        save_xyzrgb_to_ply(
            points=pred_pts[pred_mask],
            rgb=scene['images'][pred_mask],
            filename=os.path.join(args.output_dir, 'ggpt_points.ply'))
        Print(f"Saved GGPT refined points to {os.path.join(args.output_dir, 'ggpt_points.ply')}")

    Print("Pipeline complete!")


if __name__ == "__main__":
    main()
