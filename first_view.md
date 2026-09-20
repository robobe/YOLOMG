# First view: CUDA video detection

## What changed

- Added CUDA-aware inference: `dualdetector.py` and `video_detector.py` automatically use CUDA when it is available. Use `--device cpu` only to force CPU inference.
- Added `cuda_test.py` to verify that PyTorch can use the NVIDIA GPU.
- Added `video_detector.py`: choose an ARD100 video, run detection frame-by-frame, draw drone boxes and inference FPS, display the result, and save an annotated MP4.
- Updated the legacy YOLO checkpoint loader for modern PyTorch. It explicitly loads the repository's trusted local `.pt` checkpoint with `weights_only=False`.
- Replaced the removed `pkg_resources` usage so current `setuptools` versions do not prevent YOLOMG from starting.
- Added bbox rendering and image output to `dualdetector.py`.

## Installed environment

The project virtual environment now has a CUDA PyTorch build:

```text
PyTorch: 2.14.0+cu130
CUDA: 13.0
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

The NVIDIA driver is `580.173.02`. The CUDA build also runs on CPU automatically when a GPU is not available.

## Test CUDA

From the repository root:

```bash
uv run python cuda_test.py
```

Expected output includes:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
test: 2.0
```

## Run a single image demo

```bash
uv run python dualdetector.py
```

This uses the bundled checkpoint and example appearance/motion-mask pair. It saves an annotated image to `runs/detect/dualdetector.jpg`.

## Run the video detector

Open the ARD100 video picker (starts in `/home/user/datasets/ARD100/train_videos`):

```bash
uv run python video_detector.py
```

The window shows bounding boxes and inference FPS. Press `q` or `Esc` to stop. The output is saved as:

```text
runs/detect/<video-name>_detected.mp4
```

Run a known video directly, without opening the picker:

```bash
uv run python video_detector.py \
  --source /home/user/datasets/ARD100/train_videos/phantom09.mp4
```

For a headless run that only writes the output video:

```bash
uv run python video_detector.py \
  --source /home/user/datasets/ARD100/train_videos/phantom09.mp4 \
  --no-view
```

`video_detector.py` creates a simple motion image from consecutive frames because YOLOMG needs both the video frame and a motion-mask input.
