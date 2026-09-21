# YOLOMG inference

YOLOMG detects drones from two inputs:

```text
RGB image + motion-mask image -> drone boxes
```

The bundled model is:

```text
runs/train/ARD100_mask32-1280_uavs/weights/best.pt
```

## Requirements

Run commands from the repository root with the local `uv` environment.

Check that CUDA is available:

```bash
uv run python cuda_test.py
```

Expected output includes:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

YOLOMG automatically uses CUDA when it is available. Add `--device cpu` to any command to force CPU inference.

## Image inference

The included demo uses a matched RGB image and motion mask:

```bash
uv run python dualdetector.py
```

It prints the detected class, confidence, and box coordinates. The annotated result is saved to:

```text
runs/detect/dualdetector.jpg
```

Open the annotated image after inference:

```bash
uv run python dualdetector.py --view
```

Run your own RGB/mask pair:

```bash
uv run python dualdetector.py \
  --image /path/to/rgb.jpg \
  --mask /path/to/motion_mask.jpg \
  --output runs/detect/result.jpg
```

The RGB and mask images must show the same frame at the same resolution.

## Video inference

ARD100 videos are located at:

```text
/home/user/datasets/ARD100/train_videos
```

Open a file picker in that folder:

```bash
uv run python video_detector.py
```

Or run a video directly:

```bash
uv run python video_detector.py \
  --source /home/user/datasets/ARD100/train_videos/phantom09.mp4
```

The application draws drone boxes and inference FPS. Press `q` or `Esc` to stop. It writes the result to:

```text
runs/detect/<video-name>_detected.mp4
```

For a headless run without an OpenCV window:

```bash
uv run python video_detector.py \
  --source /home/user/datasets/ARD100/train_videos/phantom09.mp4 \
  --no-view
```

## Motion-mask note

The original YOLOMG dataset uses camera-motion-compensated three-frame masks. `dualdetector.py` expects such a precomputed mask as its second input.

The current video helper creates a simple consecutive-frame difference so it can run directly from a video. For a faithful benchmark of the original method, use the original compensated-mask pipeline in `test_code/FD5_mask.py`.

Read [motion_explained.md](desgin/motion_explained.md) for a beginner-friendly explanation of the motion input.

## More documentation

- [Installation and changes](desgin/installation_and_changes.md)
- [Repository guide](desgin/know_your_repo.md)
- [YOLO26 migration plan](desgin/yolo26_migration_plan.md)
