import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import cv2
import requests
import random
import numpy as np
import json

def get_dicts():
    with open('dataset.json') as json_file:
        data = json.load(json_file)
    return data

DatasetCatalog.register("can_train", get_dicts)
# DatasetCatalog.register("can_val", get_dicts)
MetadataCatalog.get("can_train").set(thing_classes=["Sticker", "Edding", "Verschmutzung"])
# MetadataCatalog.get("can_val").set(thing_classes=["Sticker", "Edding", "Verschmutzung"])
can_metadata = MetadataCatalog.get("can_train")

dataset_dicts = get_dicts()

cfg = get_cfg()
# below path applies to current installation location of Detectron2
# TODO create own config
cfgFile = "/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
cfg.merge_from_file(cfgFile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("can_val", )
predictor = DefaultPredictor(cfg)

im = cv2.imread('/test/A_01_03_06_2019315_175348.png')
# make prediction
output = predictor(im)
print(output)
