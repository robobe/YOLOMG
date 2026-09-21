# Installation and code-change summary

## Environment installed

The project uses a local `uv` virtual environment (`.venv`) with CUDA-enabled PyTorch:

```text
PyTorch: 2.14.0+cu130
CUDA runtime: 13.0
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
NVIDIA driver: 580.173.02
```

Install or refresh the CUDA PyTorch packages with:

```bash
uv pip install --reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

The exact resolved build may use a newer CUDA wheel, as in the current `2.14.0+cu130` environment.

Verify GPU support:

```bash
uv run python cuda_test.py
```

Expected results include `CUDA available: True`, the RTX 5070 GPU name, and `test: 2.0`.

## Compatibility patches

- `models/experimental.py` explicitly loads the trusted local YOLO checkpoint with `weights_only=False`. Modern PyTorch defaults to safe weight-only loading, which cannot load this legacy full-model YOLOv5 checkpoint.
- `utils/general.py` and `utils/loggers/__init__.py` no longer depend on the removed `pkg_resources` module. They use standard packaging and metadata APIs instead.
- `requirements.txt` records the legacy `setuptools<81` compatibility constraint. The source patch above remains the durable fix for current environments.

## Detector changes

### `dualdetector.py`

- Resolves checkpoints and example data relative to the script, not the shell working directory.
- Selects CUDA automatically when available; use `--device cpu` to override it.
- Checks that both RGB and motion-mask images loaded correctly.
- Scales predicted boxes to the real input image size instead of assuming 1920×1080.
- Draws boxes and labels, then saves an annotated image.

Run the image demo:

```bash
uv run python dualdetector.py
```

Useful options:

```bash
uv run python dualdetector.py --device cpu
uv run python dualdetector.py --view
uv run python dualdetector.py --output runs/detect/result.jpg
```

### `video_detector.py`

- Opens a file picker rooted at `/home/user/datasets/ARD100/train_videos`.
- Accepts a direct video path through `--source`.
- Creates a simple consecutive-frame motion input, runs the dual-input detector, overlays drone boxes and inference FPS, and writes an MP4 under `runs/detect/`.
- Press `q` or `Esc` to stop the OpenCV display.

Run with the picker:

```bash
uv run python video_detector.py
```

Run one video directly:

```bash
uv run python video_detector.py \
  --source /home/user/datasets/ARD100/train_videos/phantom09.mp4
```

Run without an OpenCV window:

```bash
uv run python video_detector.py \
  --source /home/user/datasets/ARD100/train_videos/phantom09.mp4 \
  --no-view
```

## Added documentation

- `first_view.md`: quick-start setup and commands.
- `desgin/motion_explained.md`: beginner-friendly explanation of RGB/motion inputs and inference.
- `desgin/yolo26_migration_plan.md`: the planned YOLOMG-to-YOLO26 migration.

## Git setup

The project was reinitialized as a private repository owned by `robobe`:

```text
https://github.com/robobe/YOLOMG
```

`.gitignore` excludes local environments, Python caches, editor files, and generated files under `runs/detect/`.
