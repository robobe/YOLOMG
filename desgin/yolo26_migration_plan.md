# YOLOMG dual-input YOLO26 migration plan

## Goal

Port YOLOMG to a custom dual-input YOLO26n detector while preserving the RGB-plus-motion fusion method. This is a retraining project, not a direct checkpoint conversion: the current checkpoint is a custom, anchor-based YOLOv5 derivative, whereas YOLO26 has different modules and an anchor-free head.

The baseline remains the current dual-input YOLOv5 model. The first YOLO26 version must match or exceed its mAP50-95 and improve end-to-end latency on the same held-out ARD100 videos at 1280 pixels.

## Current model

The current model accepts two three-channel tensors:

1. The RGB video frame.
2. A motion image generated from neighboring video frames.

Two small stems process the inputs. `Concat3` then applies spatial and channel attention to fuse the RGB and motion features. The fused features pass through a YOLOv5-style backbone, FPN/PAN neck, and anchor-based detection head for the single `Drone` class.

## YOLO26 architecture approach

- Keep stock YOLO26n RGB backbone, neck, and standard NMS detection head.
- Add a lightweight motion stem whose feature shape matches the first compatible YOLO26 RGB feature level.
- Port the existing attention fusion at that early feature seam.
- Expose the custom model as `forward(rgb, motion_mask)`.
- Load compatible pretrained `yolo26n.pt` tensors into unchanged RGB/neck/head layers. Motion and fusion layers start randomly initialized.
- Do not copy YOLOv5 anchors, `Detect`, loss, or NMS code. Use Ultralytics YOLO26 training and standard-NMS inference.
- Keep the implementation isolated in a dedicated `yolo26` module so the legacy baseline remains runnable.

## Motion-mask method

The original `mask32` generator is not a simple previous-frame difference. For a center frame `t`, `FD5_mask` uses three frames:

```text
t-2  ── camera-motion compensation ──>  t  <── camera-motion compensation ──  t+2
             residual 1                 |                residual 2
                                         v
                              average(residual 1, residual 2)
```

For each outer frame, grid-based KLT optical flow estimates feature movement. RANSAC derives a homography that warps that outer frame into the center frame's camera view. The absolute residual after warping highlights independently moving pixels such as the target drone, while suppressing most camera motion and background shift. The two residuals are averaged to reduce one-direction artifacts.

For the benchmark implementation, preserve this centered three-frame mask because it reproduces the original model input. The video application buffers two future frames, creating a two-frame display/output latency. A past-only causal mask is a separate future ablation and requires retraining because it changes the input distribution.

The motion implementation must become a pure reusable function: it returns a three-channel mask and has no hard-coded paths or image-writing side effects. The same function is used by dataset materialization and video inference.

## Dataset and training

- Use the raw ARD100 videos plus XML annotations in `/home/user/datasets/ARD100`.
- Preserve the original video-disjoint train/validation/test split; never randomly split frames.
- For each annotated center frame, store the RGB image, its centered motion mask, and its YOLO-format `Drone` labels.
- Apply identical resize, crop, flip, and geometric transforms to RGB, mask, and boxes. Apply color-only augmentation to RGB, never the motion mask.
- Train dual-input YOLO26n at 1280 with pretrained RGB weights, then train all layers.

## Validation and delivery

- Compare baseline and YOLO26 on the unchanged held-out videos.
- Report mAP50-95, mAP50, precision, recall, and end-to-end per-frame latency.
- Add a YOLO26 video application rooted at `/home/user/datasets/ARD100/train_videos`; it renders boxes and inference FPS and writes an annotated MP4.
- Include checks for deterministic motion-mask generation, paired RGB/mask/label integrity, video-disjoint splits, CUDA forward/backward execution, and an end-to-end annotated-video smoke test.
