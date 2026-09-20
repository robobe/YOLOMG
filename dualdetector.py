
import cv2

import argparse
import time
from pathlib import Path
import numpy as np
from numpy import random

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


def draw_predictions(img, label, score, box, color=(156, 39, 176), location=None):
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.5
    f_thickness, l_thickness = 1, 2
    
    h, w, _ = img.shape
    u1, v1, u2, v2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (u1, v1), (u2, v2), color, l_thickness)
    
    text = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
    text_h += 6
    if v1 - text_h < 0:
        cv2.rectangle(img, (u1, text_h), (u1 + text_w, 0), color, -1)
        cv2.putText(img, text, (u1, text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h), color, -1)
        cv2.putText(img, text, (u1, v1 - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    if location is not None:
        text = '(%.1fm, %.1fm)' % (location[0], location[1])
        text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
        text_h += 6
        if v2 + text_h > h:
            cv2.rectangle(img, (u1, h - text_h), (u1 + text_w, h), color, -1)
            cv2.putText(img, text, (u1, h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (u1, v2), (u1 + text_w, v2 + text_h), color, -1)
            cv2.putText(img, text, (u1, v2 + text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    
    return img

    
ROOT = Path(__file__).resolve().parent


class Yolov5Detector():
    def __init__(self, weights='', device=''):
        imgsz = 1280
        self.device = device = select_device(device)
        self.half = half = device.type != 'cpu' # half precision only supported on CUDA
        
        # Load model
        weights = Path(weights)
        weights = weights if weights.is_absolute() else ROOT / weights
        if not weights.is_file():
            raise FileNotFoundError(f'Weights not found: {weights}')
        self.model = model = attempt_load(str(weights), map_location=device, fuse=False) # load FP32 model
        self.stride = stride = int(model.stride.max()) # model stride
        self.imgsz = imgsz = check_img_size(imgsz, s=stride) # check img_size
        if half:
            model.half() # to FP16
        
        # Get names
        self.names = model.module.names if hasattr(model, 'module') else model.names
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())),torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once

    def imgdeal(self,img):
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() # uint8 to fp16/32
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def run(self, img1, img2, conf_thres=0.1, iou_thres=0.4, classes=None):
        if img1 is None or img2 is None:
            raise ValueError('Both appearance and motion-mask images must be readable.')
        img0_shape = img1.shape[:2]
        # Padded resize
        img1 = self.imgdeal(img1)
        img2 = self.imgdeal(img2)
        # print(img1.shape)
        # Inference
        pred = self.model(img1, img2, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)
        
        # Process detections
        det = pred[0]

        if len(det):
            boxes = scale_coords(img1.shape[2:], det[:, :4], img0_shape).round().cpu().numpy() # xyxy
            labels = [self.names[int(cls)] for cls in det[:, -1]]
            scores = [float('%.2f' % conf) for conf in det[:, -2]]
            return labels, scores, boxes
        else:
            return [], [], np.array([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLOMG on an image and its motion mask.')
    parser.add_argument('--weights', default='runs/train/ARD100_mask32-1280_uavs/weights/best.pt')
    parser.add_argument('--image', default='data/Test_images/images/phantom05_0606.jpg')
    parser.add_argument('--mask', default='data/Test_images/mask/phantom05_0606.jpg')
    parser.add_argument('--output', default='runs/detect/dualdetector.jpg')
    parser.add_argument('--view', action='store_true', help='Display the annotated image.')
    parser.add_argument('--device', default='', help="CUDA device, e.g. '0', or 'cpu' (default: CUDA if available)")
    args = parser.parse_args()

    detector = Yolov5Detector(weights=args.weights, device=args.device)
    image_path = Path(args.image) if Path(args.image).is_absolute() else ROOT / args.image
    mask_path = Path(args.mask) if Path(args.mask).is_absolute() else ROOT / args.mask
    img1 = cv2.imread(str(image_path))
    img2 = cv2.imread(str(mask_path))
    
    t1 = time.time()
    labels, scores, boxes = detector.run(img1, img2)
    t2 = time.time()
    print('time cost:', t2 - t1, '\n')
    
    print('labels: ', labels)
    print('scores: ', scores)
    print('boxes: ', boxes)

    for label, score, box in zip(labels, scores, boxes):
        draw_predictions(img1, label, score, box)
    output_path = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), img1):
        raise OSError(f'Could not write output image: {output_path}')
    print('output:', output_path)
    if args.view:
        cv2.imshow('YOLOMG', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
