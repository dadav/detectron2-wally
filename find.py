#!/usr/bin/env python3

import os
import sys
import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog


# vars
INPUT_FILE = sys.argv[1]
MODEL_PATH = os.path.realpath('output/model_final.pth')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = MODEL_PATH
if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

im = cv2.imread(INPUT_FILE)
outputs = predictor(im)

instances = outputs['instances']


if len(instances) <= 0:
    sys.exit(1)

v = Visualizer(im[:, :, ::-1],
               MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
               scale=1.2,
               instance_mode=ColorMode.IMAGE_BW)

v = v.draw_instance_predictions(instances.to("cpu"))

result = v.get_image()[:, :, ::-1]

cv2.imshow('waldo', result)
while True:
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break
cv2.destroyAllWindows()
