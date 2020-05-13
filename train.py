# -*- coding: utf-8 -*-
import os
import torch, torchvision

# Basic setup
# log
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import json
import itertools
from detectron2.structures import BoxMode


def csv_to_list_of_dicts(path):
  import csv
  with open(path, mode='r') as infile:
      reader = csv.DictReader(infile)
      return list(reader)


def get_wally_dicts(img_dir):
    annotations = csv_to_list_of_dicts(os.path.join(os.path.dirname(img_dir), 'annotations.csv'))
    files = os.listdir(img_dir)

    dataset_dicts = []
    for idx, v in enumerate(annotations):
      if v["filename"] not in files:
        continue
      record = {}

      filename = os.path.join(img_dir, v["filename"])
      height, width = cv2.imread(filename).shape[:2]

      record["file_name"] = filename
      record["image_id"] = idx
      record["height"] = height
      record["width"] = width
      xmin = int(v['xmin'])
      ymin = int(v['ymin'])
      xmax = int(v['xmax'])
      ymax = int(v['ymax'])
      poly = [
        (xmin, ymin), (xmax, ymin),
        (xmax, ymax), (xmin, ymax)
      ]
      poly = list(itertools.chain.from_iterable(poly))
      px = [xmin, xmax]
      py = [int(v['ymin']), int(v['ymax'])]
      obj = {
          "bbox": [np.int64(xmin), np.int64(ymin), np.int64(xmax), np.int64(ymax)],
          "bbox_mode": BoxMode.XYXY_ABS,
          "segmentation": [poly],
          "category_id": 0,
          "iscrowd": 0
      }
      record["annotations"] = [obj]
      dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog

DatasetCatalog.clear()

#for d in ["train", "val"]:
for d in ["train"]:
  DatasetCatalog.register("wally_images_" + d, lambda d=d: get_wally_dicts("images/" + d))
  MetadataCatalog.get("wally_images_" + d).set(thing_classes=["wally"])

wally_metadata = MetadataCatalog.get("wally_images_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("wally_images_train",)
cfg.DATASETS.TEST = () #("wally_images_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 600    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

