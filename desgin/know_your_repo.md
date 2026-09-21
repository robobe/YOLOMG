# Know your YOLOMG repository

## What this project is

YOLOMG is a modified YOLOv5 project for finding drones in drone videos. Its special idea is that it uses two images for every prediction:

```text
normal RGB frame + motion-mask frame -> drone bounding box
```

The original project includes the model code, training code, evaluation code, sample images/masks, saved model weights, and scripts for making the motion-mask dataset.

## Main files

| File or folder | What it is for |
| --- | --- |
| `README.md` | Original project overview, dataset link, and basic train/validation commands. |
| `train.py` | Trains the original dual-input YOLOMG model. |
| `val.py` | Evaluates a trained model and calculates detection metrics. |
| `export.py` | Exports a PyTorch model to deployment formats such as ONNX. |
| `dualdetector.py` | Runs one RGB image and one motion-mask image through the model. |
| `models/` | Model architecture code and YAML model descriptions. |
| `utils/` | Shared YOLOv5 helpers: dataset loading, box math, loss, plotting, CUDA setup, and logging. |
| `data/` | Dataset YAML files, label-conversion scripts, hyperparameters, and example RGB/mask pairs. |
| `test_code/` | Original ARD100 preparation scripts: extract frames, generate masks, convert annotations, and test detection. |
| `runs/train/` | Original saved training results and `best.pt` model checkpoints. |

## The important original model files

### `models/dual_uav2.yaml`

This describes the dual-input network structure. It declares two input branches:

- A small branch for the normal RGB video image.
- A small branch for the motion image.

The `Concat3` layer joins the two branches with attention. After that, the model works much like a normal YOLOv5 detector.

### `models/yolo.py`

This is the model builder and forward pass. Unlike ordinary YOLOv5, its forward function receives two tensors:

```python
model(rgb_image, motion_mask)
```

### `models/common.py`

Contains reusable layers. The special YOLOMG layer is `Concat3`, which uses spatial and channel attention to combine RGB features with motion features.

### `models/experimental.py`

Loads `.pt` checkpoints. It was updated to load the original trusted checkpoint correctly with modern PyTorch.

## The original dataset workflow

```text
ARD100 videos + XML annotations
        |
        +-- extract video frames
        |
        +-- create camera-motion-compensated masks
        |
        +-- convert XML boxes to YOLO labels
        |
        v
paired RGB image, motion mask, and drone label
        |
        v
train.py / val.py
```

Useful original scripts in `test_code/`:

| Script | Purpose |
| --- | --- |
| `generate_mask5.py` | Creates the original three-frame `mask32` motion images. |
| `FD5_mask.py` | Calculates one motion mask from neighboring frames. |
| `MOD_Functions.py` | Contains optical-flow and homography camera-motion compensation. |
| `YOLOMG_extract_frames.py` | Extracts video frames. |
| `generate_dataset.py` | Builds dataset file lists and labels. |
| `yolov5_dualdetector.py` | Older example detector script from the original project. |

## New helper files added in this copy

| File | Purpose |
| --- | --- |
| `cuda_test.py` | Confirms that PyTorch can use the NVIDIA GPU. |
| `video_detector.py` | Lets you select an ARD100 video, runs detection, draws boxes/FPS, and saves an MP4. |
| `first_view.md` | Quick-start commands. |
| `desgin/installation_and_changes.md` | CUDA installation and change summary. |
| `desgin/motion_explained.md` | Beginner explanation of the motion input. |
| `desgin/yolo26_migration_plan.md` | Planned migration from YOLOMG to dual-input YOLO26. |

## Normal usage today

Test CUDA:

```bash
uv run python cuda_test.py
```

Run the example image and motion mask:

```bash
uv run python dualdetector.py
```

Select and process a video:

```bash
uv run python video_detector.py
```

The example detector uses `runs/train/ARD100_mask32-1280_uavs/weights/best.pt` by default.
