"""Choose an ARD100 video, detect drones, and save an annotated copy."""

import argparse
import csv
import re
import time
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk
from xml.etree import ElementTree

import cv2
import numpy as np
import yaml

VIDEO_DIR = Path('/home/user/datasets/ARD100/train_videos')
FRAME_NUMBER = re.compile(r'.*_(\d+)$')
ROOT = Path(__file__).resolve().parent
PRESETS_FILE = ROOT / 'video_detector_presets.yaml'
IOU_THRESHOLD = 0.5


def load_presets(path=PRESETS_FILE):
    """Load named video/annotation pairs from YAML."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid presets YAML: {path}') from error
    if not isinstance(document, dict):
        raise ValueError(f'Invalid presets YAML: {path}')
    presets = document.get('presets', {})
    if not isinstance(presets, dict) or any(
        not isinstance(name, str) or not isinstance(value, dict) or not isinstance(value.get('video'), str)
        or value.get('annotations') is not None and not isinstance(value['annotations'], str)
        for name, value in presets.items()
    ):
        raise ValueError(f'Invalid presets YAML: {path}')
    return presets


def save_presets(presets, path=PRESETS_FILE):
    Path(path).write_text(yaml.safe_dump({'presets': presets}, sort_keys=False))


def load_annotations(directory, frame_count, width, height):
    """Load Pascal-VOC boxes keyed by one-based video frame number."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Annotation directory does not exist: {directory}')

    annotations = {}
    for path in sorted(directory.glob('*.xml')):
        match = FRAME_NUMBER.fullmatch(path.stem)
        if not match:
            raise ValueError(f'Cannot read frame number from annotation: {path}')
        frame_number = int(match.group(1))
        if not 1 <= frame_number <= frame_count:
            raise ValueError(f'Annotation frame {frame_number} is outside video range 1-{frame_count}: {path}')
        if frame_number in annotations:
            raise ValueError(f'Duplicate annotation for frame {frame_number}: {path}')

        try:
            root = ElementTree.parse(path).getroot()
            size = root.find('size')
            xml_width = int(size.findtext('width'))
            xml_height = int(size.findtext('height'))
        except (AttributeError, TypeError, ValueError, ElementTree.ParseError) as error:
            raise ValueError(f'Invalid annotation metadata: {path}') from error
        if (xml_width, xml_height) != (width, height):
            raise ValueError(f'Annotation size {xml_width}x{xml_height} does not match video {width}x{height}: {path}')

        boxes = []
        for obj in root.findall('object'):
            label = obj.findtext('name')
            box = obj.find('bndbox')
            try:
                xmin, ymin, xmax, ymax = (int(box.findtext(name)) for name in ('xmin', 'ymin', 'xmax', 'ymax'))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f'Invalid bounding box: {path}') from error
            if not label or not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                raise ValueError(f'Out-of-bounds bounding box: {path}')
            boxes.append((label, (xmin, ymin, xmax, ymax)))
        if not boxes:
            raise ValueError(f'Annotation has no objects: {path}')
        annotations[frame_number] = boxes
    if not annotations:
        raise ValueError(f'No XML annotations found in: {directory}')
    return annotations


def draw_ground_truth(frame, label, box):
    xmin, ymin, xmax, ymax = box
    color = (0, 255, 0)
    text = f'GT: {label}'
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(frame, text, (xmin, max(20, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def box_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / area if area else 0.0


def compare_frame(truths, labels, boxes):
    """Return mean best IoU and the number of matched/missed ground-truth boxes."""
    best_ious = [max((box_iou(truth, box) for predicted, box in zip(labels, boxes) if predicted == label), default=0.0)
                 for label, truth in truths]
    matched = sum(iou >= IOU_THRESHOLD for iou in best_ious)
    return sum(best_ious) / len(best_ious), matched, len(best_ious) - matched


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


def choose_inputs():
    """Choose a video and its optional Pascal-VOC annotation directory."""
    root = Tk()
    root.title('YOLOMG video inference')
    root.columnconfigure(1, weight=1)
    try:
        presets = load_presets()
    except ValueError as error:
        root.destroy()
        raise SystemExit(error) from error
    video_name = StringVar(root)
    video_directory = StringVar(root)
    annotation_name = StringVar(root)
    annotation_directory = StringVar(root)
    preset_name = StringVar(root)
    selected = []

    def set_video(path):
        path = Path(path)
        video_directory.set(str(path.parent))
        video_name.set(path.name)

    def set_annotations(path):
        path = Path(path)
        annotation_directory.set(str(path.parent))
        annotation_name.set(path.name)

    def video_path():
        return Path(video_directory.get()) / video_name.get()

    def annotations_path():
        return Path(annotation_directory.get()) / annotation_name.get() if annotation_name.get() else None

    def browse_video():
        path = filedialog.askopenfilename(
            parent=root, title='Choose video', initialdir=VIDEO_DIR,
            filetypes=[('Videos', '*.mp4 *.avi *.mov'), ('All files', '*.*')],
        )
        if path:
            set_video(path)

    def browse_annotations():
        path = filedialog.askdirectory(parent=root, title='Choose XML annotation directory', initialdir=VIDEO_DIR.parent / 'annotations')
        if path:
            set_annotations(path)

    def load_preset(_event=None):
        preset = presets[preset_name.get()]
        set_video(preset['video'])
        if preset.get('annotations'):
            set_annotations(preset['annotations'])
        else:
            annotation_directory.set('')
            annotation_name.set('')

    def save_preset():
        name = preset_name.get().strip()
        if not name:
            messagebox.showerror('Preset name', 'Enter a preset name.', parent=root)
            return
        path = video_path()
        if not path.is_file():
            messagebox.showerror('Choose video', 'Choose a readable video file first.', parent=root)
            return
        annotation_path = annotations_path()
        if annotation_path and not annotation_path.is_dir():
            messagebox.showerror('Choose annotations', 'Choose a valid annotation directory.', parent=root)
            return
        presets[name] = {'video': str(path), 'annotations': str(annotation_path) if annotation_path else None}
        save_presets(presets)
        preset_picker['values'] = list(presets)

    def run():
        path = video_path()
        if not path.is_file():
            messagebox.showerror('Choose video', 'Choose a readable video file.', parent=root)
            return
        annotation_path = annotations_path()
        presets['Last selection'] = {'video': str(path), 'annotations': str(annotation_path) if annotation_path else None}
        save_presets(presets)
        selected.extend((path, annotation_path))
        root.destroy()

    ttk.Label(root, text='Preset').grid(row=0, column=0, padx=8, pady=8, sticky='w')
    preset_picker = ttk.Combobox(root, textvariable=preset_name, values=list(presets))
    preset_picker.grid(row=0, column=1, padx=8, pady=8, sticky='ew')
    preset_picker.bind('<<ComboboxSelected>>', load_preset)
    if 'Last selection' in presets:
        preset_name.set('Last selection')
        load_preset()
    ttk.Label(root, text='Video').grid(row=1, column=0, padx=8, pady=8, sticky='w')
    ttk.Entry(root, textvariable=video_name).grid(row=1, column=1, padx=8, pady=8, sticky='ew')
    ttk.Button(root, text='Choose video', command=browse_video).grid(row=1, column=2, padx=8, pady=8)
    ttk.Label(root, text='Video path').grid(row=2, column=0, padx=8, pady=(0, 8), sticky='w')
    ttk.Label(root, textvariable=video_directory).grid(row=2, column=1, columnspan=2, padx=8, pady=(0, 8), sticky='w')
    ttk.Label(root, text='Annotations').grid(row=3, column=0, padx=8, pady=8, sticky='w')
    ttk.Entry(root, textvariable=annotation_name).grid(row=3, column=1, padx=8, pady=8, sticky='ew')
    ttk.Button(root, text='Choose folder', command=browse_annotations).grid(row=3, column=2, padx=8, pady=8)
    ttk.Label(root, text='Annotation path').grid(row=4, column=0, padx=8, pady=(0, 8), sticky='w')
    ttk.Label(root, textvariable=annotation_directory).grid(row=4, column=1, columnspan=2, padx=8, pady=(0, 8), sticky='w')
    ttk.Button(root, text='Save preset', command=save_preset).grid(row=5, column=1, padx=8, pady=(0, 8), sticky='e')
    ttk.Button(root, text='Run inference', command=run).grid(row=5, column=2, padx=8, pady=(0, 8), sticky='e')
    root.mainloop()
    if not selected:
        raise SystemExit('No video selected.')
    return selected


def motion_mask(previous, current):
    """Return the current simple motion input as a three-channel image.

    This is an absolute difference between consecutive blurred grayscale frames.
    Bright pixels mean that the same screen location changed between frames.

    Important: this helper does *not* ignore camera movement. If the camera
    pans, stationary buildings also move on screen and can become bright.

    The original YOLOMG mask32 pipeline removes most global camera movement:
      1. Track a grid of background points with KLT optical flow.
      2. Use RANSAC to fit a background homography and reject outlier tracks
         from drones, cars, or bad matches.
      3. Warp the neighboring frame into the current camera view.
      4. Subtract the warped frame from the current frame. Background should
         align and become dark; independently moving drones stay bright.

    That original three-frame implementation is in test_code/FD5_mask.py and
    test_code/MOD_Functions.py. It should replace this simple helper for a
    faithful YOLOMG benchmark, using frames t-2, t, and t+2.
    """
    previous = cv2.cvtColor(cv2.GaussianBlur(previous, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(cv2.GaussianBlur(current, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.absdiff(previous, current), cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description='Run YOLOMG on an ARD100 video.')
    parser.add_argument('--source', help='Video file, or directory to open in the picker.')
    parser.add_argument('--weights', default='runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--device', default='', help="CUDA device, e.g. '0', or 'cpu' (default: CUDA if available)")
    parser.add_argument('--annotations', help='Pascal-VOC XML directory to overlay as ground truth.')
    parser.add_argument('--no-view', action='store_true', help='Save without opening an OpenCV window.')
    args = parser.parse_args()

    from dualdetector import Yolov5Detector, draw_predictions

    detector = Yolov5Detector(args.weights, args.device)

    if args.source:
        source = choose_video(args.source)
        annotations_path = args.annotations
    else:
        source, annotations_path = choose_inputs()
        annotations_path = args.annotations or annotations_path
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f'Could not open video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotations = load_annotations(annotations_path, frame_count, width, height) if annotations_path else {}
    output = ROOT / 'runs' / 'detect' / f'{source.stem}_detected.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f'Could not create video: {output}')
    summary_path = output.with_name(f'{source.stem}_summary.csv') if annotations else None
    summary_file = summary_path.open('w', newline='') if summary_path else None
    summary_writer = csv.DictWriter(summary_file, fieldnames=('frame', 'ground_truths', 'predictions', 'best_iou', 'matched', 'failures')) if summary_file else None
    if summary_writer:
        summary_writer.writeheader()
    total_truths = total_matched = total_failures = 0
    total_iou = 0.0

    previous = None
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        # The first frame has no neighbor, so use an empty mask. Later frames
        # use the previous raw frame; store it before drawing boxes so overlays
        # never appear as false motion in the next mask.
        mask = np.zeros_like(frame) if previous is None else motion_mask(previous, frame)
        previous = frame.copy()
        started = time.perf_counter()
        labels, scores, boxes = detector.run(frame, mask)
        inference_fps = 1 / (time.perf_counter() - started)
        for label, score, box in zip(labels, scores, boxes):
            draw_predictions(frame, label, score, box)
        truths = annotations.get(frame_number, [])
        for label, box in truths:
            draw_ground_truth(frame, label, box)
        if truths:
            best_iou, matched, failures = compare_frame(truths, labels, boxes)
            summary_writer.writerow({
                'frame': frame_number, 'ground_truths': len(truths), 'predictions': len(boxes),
                'best_iou': f'{best_iou:.6f}', 'matched': matched, 'failures': failures,
            })
            total_truths += len(truths)
            total_matched += matched
            total_failures += failures
            total_iou += best_iou * len(truths)
        cv2.putText(frame, f'Inference: {inference_fps:.1f} FPS', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(frame)
        if not args.no_view:
            cv2.imshow('YOLOMG', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    capture.release()
    writer.release()
    if summary_file:
        summary_file.close()
    cv2.destroyAllWindows()
    print(f'output: {output}')
    if summary_path:
        print(f'summary: {summary_path}')
        print(f'ground truth={total_truths}, matched={total_matched}, failures={total_failures}, mean IoU={total_iou / total_truths:.3f}')


if __name__ == '__main__':
    main()
