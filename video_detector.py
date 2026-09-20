"""Choose an ARD100 video, detect drones, and save an annotated copy."""

import argparse
import time
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np

from dualdetector import ROOT, Yolov5Detector, draw_predictions


VIDEO_DIR = Path('/home/user/datasets/ARD100/train_videos')


def choose_video(source):
    source = Path(source)
    if source.is_file():
        return source
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(initialdir=source, filetypes=[('Videos', '*.mp4 *.avi *.mov')])
    root.destroy()
    if not filename:
        raise SystemExit('No video selected.')
    return Path(filename)


def motion_mask(previous, current):
    previous = cv2.cvtColor(cv2.GaussianBlur(previous, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(cv2.GaussianBlur(current, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.absdiff(previous, current), cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description='Run YOLOMG on an ARD100 video.')
    parser.add_argument('--source', default=VIDEO_DIR, help='Video file, or directory to open in the picker.')
    parser.add_argument('--weights', default='runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--device', default='', help="CUDA device, e.g. '0', or 'cpu' (default: CUDA if available)")
    parser.add_argument('--no-view', action='store_true', help='Save without opening an OpenCV window.')
    args = parser.parse_args()

    source = choose_video(args.source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f'Could not open video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = ROOT / 'runs' / 'detect' / f'{source.stem}_detected.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f'Could not create video: {output}')

    detector = Yolov5Detector(args.weights, args.device)
    previous = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = np.zeros_like(frame) if previous is None else motion_mask(previous, frame)
        previous = frame.copy()
        started = time.perf_counter()
        labels, scores, boxes = detector.run(frame, mask)
        inference_fps = 1 / (time.perf_counter() - started)
        for label, score, box in zip(labels, scores, boxes):
            draw_predictions(frame, label, score, box)
        cv2.putText(frame, f'Inference: {inference_fps:.1f} FPS', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(frame)
        if not args.no_view:
            cv2.imshow('YOLOMG', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'output: {output}')


if __name__ == '__main__':
    main()
