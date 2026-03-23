# GGPT CPU Setup & Usage Guide

## Overview

GGPT generates geometrically grounded 3D point clouds from multiview images or video frames. This guide covers setting up and running GGPT on **macOS Intel (x86_64) without GPU** using Anaconda.

The pipeline produces PLY point cloud files from 5-10 input images of a scene.

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
# From your anaconda installation (adjust path as needed)
/Users/yiliu/opt/anaconda3/bin/conda create -n ggpt_cpu python=3.10 -y
```

### 1.2 Activate & Install Dependencies

```bash
export PATH="/Users/yiliu/opt/anaconda3/envs/ggpt_cpu/bin:$PATH"
cd /Users/yiliu/Documents/Repos/GGPT

# PyTorch CPU (2.2.2 is the latest for macOS x86_64)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# numpy <2 required (PyTorch 2.2.2 incompatible with numpy 2.x)
# opencv <4.13 required (4.13+ forces numpy 2.x)
pip install "numpy<2" "opencv-python==4.10.0.84"

# Core dependencies
pip install Pillow huggingface_hub einops safetensors
pip install pycolmap==3.12.0
pip install git+https://github.com/cvg/LightGlue.git
pip install hydra-core h5py pyyaml scipy plyfile addict timm matplotlib

# RoMaV2 (from local submodule, patched for CPU)
cd RoMaV2 && pip install -e . && cd ..

# torch-scatter (for GGPT refinement stage)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
```

### 1.3 Download GGPT Checkpoint (for refinement mode)

```bash
export PATH="/Users/yiliu/opt/anaconda3/envs/ggpt_cpu/bin:$PATH"
mkdir -p ckpts
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='YutongGoose/GGPT', filename='model.step228000.pth', local_dir='ckpts/')
print('Checkpoint downloaded.')
"
```

The VGGT-1B model (~5GB) and RoMaV2 weights (~1GB) are downloaded automatically on first run.

### 1.4 Verify Installation

```bash
export PATH="/Users/yiliu/opt/anaconda3/envs/ggpt_cpu/bin:$PATH"
python -c "
import torch; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())
import numpy as np; print('numpy', np.__version__)
import pycolmap; print('pycolmap', pycolmap.__version__)
import romav2; print('romav2 OK')
from lightglue import SuperPoint; print('LightGlue OK')
print('All OK!')
"
```

Expected output:
```
PyTorch 2.2.2 - CUDA: False
numpy 1.26.4
pycolmap 3.12.0
romav2 OK
LightGlue OK
All OK!
```

---

## 2. Running the Pipeline

### 2.1 Entry Point

Use `run_demo_cpu.py` — a standalone runner that works on CPU without Hydra (which causes segfaults with pycolmap on macOS).

### 2.2 Basic Run (SfM-only, no GGPT refinement)

```bash
export PATH="/Users/yiliu/opt/anaconda3/envs/ggpt_cpu/bin:$PATH"
cd /Users/yiliu/Documents/Repos/GGPT

python -u run_demo_cpu.py \
    --image_dir examples/scannetpp-9bb22982672c69bf \
    --max_width 640 \
    --max_images 5 \
    --output_dir outputs/demo_cpu
```

### 2.3 Full Run (with GGPT Point Transformer refinement)

```bash
python -u run_demo_cpu.py \
    --image_dir examples/scannetpp-9bb22982672c69bf \
    --max_width 640 \
    --max_images 5 \
    --ggpt_refine \
    --output_dir outputs/demo_cpu_ggpt
```

### 2.4 Using Your Own Images

Place 5–10 images (`.jpg`, `.png`, `.jpeg`) of a scene in a directory:

```bash
python -u run_demo_cpu.py \
    --image_dir /path/to/your/images \
    --max_width 640 \
    --max_images 8 \
    --output_dir outputs/my_scene
```

### 2.5 Using Video Input

Extract frames from a video first (e.g., 1 frame per second):

```bash
python -c "
import cv2, os
vs = cv2.VideoCapture('/path/to/video.mp4')
fps = vs.get(cv2.CAP_PROP_FPS)
interval = int(fps)  # 1 frame/sec
os.makedirs('frames', exist_ok=True)
idx, saved = 0, 0
while saved < 10:
    ok, frame = vs.read()
    if not ok: break
    if idx % interval == 0:
        cv2.imwrite(f'frames/{saved:06d}.jpg', frame)
        saved += 1
    idx += 1
vs.release()
print(f'Saved {saved} frames')
"

python -u run_demo_cpu.py --image_dir frames --max_images 8 --output_dir outputs/video_scene
```

---

## 3. Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--image_dir` | `examples/scannetpp-...` | Directory containing input images |
| `--output_dir` | `outputs/demo_cpu` | Where to save PLY output files |
| `--max_width` | `640` | Max image width (lower = faster, less memory) |
| `--max_images` | `8` | Max number of images to use (5–10 recommended) |
| `--matcher` | `romav2-base` | Matching model (`romav2-base`, `romav2-fast`) |
| `--ggpt_refine` | `False` | Enable GGPT Point Transformer refinement |
| `--max_pts_num` | `1000000` | Max points in output PLY |
| `--conf_quantile_thresh` | `0.2` | Confidence filter (lower = keep more points) |

---

## 4. Output Files

Output appears in `--output_dir`:

| File | Description |
|---|---|
| `sfm_dlt_points.ply` | Sparse SfM triangulated 3D points (always produced) |
| `ff_points.ply` | Dense VGGT feedforward predictions (always produced) |
| `ggpt_points.ply` | GGPT-refined points (only with `--ggpt_refine`) |

Each PLY file contains per-vertex XYZ coordinates + RGB color. Open with MeshLab, CloudCompare, Open3D, or any PLY viewer.

### Verify output:

```bash
export PATH="/Users/yiliu/opt/anaconda3/envs/ggpt_cpu/bin:$PATH"
python -c "
from plyfile import PlyData
for f in ['sfm_dlt_points.ply', 'ff_points.ply']:
    p = PlyData.read(f'outputs/demo_cpu/{f}')
    print(f'{f}: {len(p[\"vertex\"])} points')
"
```

---

## 5. Performance & Memory Notes

| Stage | Time (5 images, 640px) | RAM |
|---|---|---|
| VGGT feedforward | ~2 min | ~8 GB |
| RoMaV2 dense matching | ~25–30 min | ~6 GB |
| SfM (BA + triangulation) | ~1 min | ~2 GB |
| GGPT refinement | ~2–5 min | ~4 GB |
| **Total (without GGPT)** | **~30 min** | **~8 GB peak** |
| **Total (with GGPT)** | **~35 min** | **~8 GB peak** |

Tips for reducing time/memory:
- Use `--max_width 518` (VGGT native resolution, avoids upsampling)
- Use `--max_images 5` instead of 8 (matching is O(n²) in image count)
- Use `--matcher romav2-fast` for faster (but less precise) matching

---

## 6. Code Modifications Summary

Files modified from original GGPT repo for CPU support:

| File | Change |
|---|---|
| `run_demo.py` | Device auto-detection, CUDA guards, ggpt_refine conditional |
| `run_demo_cpu.py` | **New** — standalone CPU runner bypassing Hydra |
| `feedforward/__init__.py` | CPU-safe dtype selection and autocast |
| `sfm/sfm_func.py` | Guard `torch.cuda.empty_cache()` |
| `utils/basic.py` | Guard CUDA seed/cudnn calls |
| `vggt/vggt/models/vggt.py` | CPU-safe autocast context manager |
| `configs/demo.yaml` | `ggpt_refine: False` default |
| `ggpt/model/base.py` | Lazy imports for SpUNetBase/SPVCNN, dict access fix |
| `spconv_shim.py` | **New** — CPU replacement for spconv (sparse conv) |
| `install_spconv_shim.py` | **New** — registers shim in sys.modules |
| `Pointcept/pointcept/models/__init__.py` | Safe imports for missing deps |
| `RoMaV2/src/romav2/device.py` | Force CPU (skip MPS) |
| `RoMaV2/src/romav2/features.py` | Disable bfloat16 autocast on CPU |
| `RoMaV2/src/romav2/matcher.py` | `nn.Buffer` → `register_buffer`, disable autocast on CPU |
| `RoMaV2/src/romav2/romav2.py` | Skip `torch.compile` on CPU |
| `RoMaV2/pyproject.toml` | Remove `dataclasses` dep, relax `torchvision` version |
| DINOv3 hub cache | Patch `custom_fwd`/`custom_bwd` for PyTorch 2.2 compat |

---

## 7. Troubleshooting

**SIGSEGV on macOS with Hydra runner (`run_demo.py`)**
- pycolmap's native library triggers a background-thread segfault on macOS when combined with Hydra's signal handlers. Use `run_demo_cpu.py` instead.

**numpy version conflict**
- PyTorch 2.2.2 (latest for macOS x86_64) requires numpy <2. opencv-python >=4.13 requires numpy >=2. Solution: use `opencv-python==4.10.0.84`.

**RoMaV2 `nn.Buffer` error**
- `nn.Buffer` requires PyTorch 2.4+. Already patched to use `register_buffer()`.

**DINOv3 `custom_fwd` / `custom_bwd` error**
- The cached DINOv3 hub code at `~/.cache/torch/hub/facebookresearch_dinov3_*/` needs patching for PyTorch 2.2. This is done automatically on first RoMaV2 load, but if the cache is cleared, re-apply the patch in `dinov3/eval/segmentation/models/utils/ms_deform_attn.py`.

**Out of memory**
- Reduce `--max_width` to 518, `--max_images` to 5, or use `--matcher romav2-fast`.
