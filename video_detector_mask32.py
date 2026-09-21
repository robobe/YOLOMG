"""Run YOLOMG with the original three-frame mask32 motion pipeline."""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from dualdetector import ROOT, Yolov5Detector, draw_predictions
from video_detector import VIDEO_DIR, choose_video


def compensated_frame(source, target):
    """Warp source into target's camera view using KLT tracks and RANSAC."""
    height, width = target.shape
    source_small = cv2.resize(source, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    target_small = cv2.resize(target, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    points = np.array([(x, y) for x in range(32, 1856, 64) for y in range(24, 1032, 48)], dtype=np.float32).reshape(-1, 1, 2)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        source_small, target_small, points, None,
        winSize=(15, 15), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
    )
    if tracked is None or status is None:
        return source
    source_points = points[status.ravel() == 1]
    target_points = tracked[status.ravel() == 1]
    if len(source_points) < 15:
        return source
    homography, _ = cv2.findHomography(target_points, source_points, cv2.RANSAC, 3.0)
    if homography is None:
        return source
    return cv2.warpPerspective(source, homography, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def mask32(earlier, target, later):
    """Return the original YOLOMG-style mask for target from t-2, t, and t+2.

    Each outer frame is aligned to target first. This cancels shared camera
    motion, so stationary background becomes dark after subtraction while a
    drone that moves differently remains bright.
    """
    frames = [cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2GRAY) for frame in (earlier, target, later)]
    previous, target, future = frames
    residual_previous = cv2.absdiff(target, compensated_frame(previous, target))
    residual_future = cv2.absdiff(target, compensated_frame(future, target))
    return cv2.cvtColor(cv2.addWeighted(residual_previous, 0.5, residual_future, 0.5, 0), cv2.COLOR_GRAY2BGR)


def self_check():
    """Check that compensation reduces a synthetic global camera translation."""
    source = np.random.default_rng(0).integers(0, 256, (240, 320), dtype=np.uint8)
    target = cv2.warpAffine(source, np.float32([[1, 0, 12], [0, 1, 6]]), (320, 240))
    aligned = compensated_frame(source, target)
    assert cv2.absdiff(target, aligned).mean() < cv2.absdiff(target, source).mean()
    print('mask32 camera-compensation check: ok')


def main():
    parser = argparse.ArgumentParser(description='Run YOLOMG with the original mask32 motion pipeline.')
    parser.add_argument('--source', default=VIDEO_DIR, help='Video file, or directory to open in the picker.')
    parser.add_argument('--weights', default='runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--device', default='', help="CUDA device, e.g. '0', or 'cpu' (default: CUDA if available)")
    parser.add_argument('--no-view', action='store_true', help='Save without opening an OpenCV window.')
    parser.add_argument('--self-check', action='store_true', help='Verify camera-motion compensation, then exit.')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    source = choose_video(args.source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f'Could not open video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = ROOT / 'runs' / 'detect' / f'{source.stem}_mask32_detected.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f'Could not create video: {output}')

    detector = Yolov5Detector(args.weights, args.device)
    frames = deque(maxlen=5)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) < 5:
            continue

        # The center frame is two frames behind the newest frame. This
        # two-frame delay is required to reproduce the original mask32 input.
        earlier, target, later = frames[0], frames[2], frames[4]
        started = time.perf_counter()
        labels, scores, boxes = detector.run(target, mask32(earlier, target, later))
        inference_fps = 1 / (time.perf_counter() - started)
        for label, score, box in zip(labels, scores, boxes):
            draw_predictions(target, label, score, box)
        cv2.putText(target, f'Inference: {inference_fps:.1f} FPS', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(target)
        if not args.no_view:
            cv2.imshow('YOLOMG mask32', target)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'output: {output}')


if __name__ == '__main__':
    main()
