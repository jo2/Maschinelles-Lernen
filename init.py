import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import requests
import random
import numpy as np
import json
import os
from vovnet import add_vovnet_config

from detectron2.data.datasets import register_coco_instances

import torch
torch.cuda.is_available()
print(torch.cuda.is_available())

def get_dicts():
    with open('/volume/dataset.json') as json_file:
        data = json.load(json_file)
    return data

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

clear_folder("/volume/processed")
clear_folder("/volume/evaluated")

register_coco_instances("can_train", {}, "/volume/dataset.json", "/volume/img")
register_coco_instances("can_val", {}, "/volume/dataset.json", "/volume/test")

dataset_dicts = DatasetCatalog.get("can_train")
can_metadata = MetadataCatalog.get("can_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=can_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    processedImg = vis.get_image()[:, :, ::-1]
    cv2.imwrite("/volume/processed/" + d["file_name"][12:], processedImg)

cfg = get_cfg()
add_vovnet_config(cfg)
cfg.merge_from_file("/volume/configs/centermask_V_39_eSE_FPN_ms_3x.yaml")
# cfg.merge_from_file("/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("can_train")
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.MODEL.DEVICE = "cpu"
cfg.SOLVER.MAX_ITER = (30)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print("finished training")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("can_val")
predictor = DefaultPredictor(cfg)

dataset_val = DatasetCatalog.get("can_val")

for d in random.sample(dataset_val["images"], 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("/volume/evaluated/" + d["file_name"][13:], v.get_image()[:, :, ::-1])




# DatasetCatalog.register("can_train", get_dicts)
# DatasetCatalog.register("can_val", get_dicts)
# MetadataCatalog.get("can_train").set(thing_classes=["Sticker", "Edding", "Verschmutzung"])
# MetadataCatalog.get("can_val").set(thing_classes=["Sticker", "Edding", "Verschmutzung"])
# can_metadata = MetadataCatalog.get("can_train")

# dataset_dicts = get_dicts()

# cfg = get_cfg()
# below path applies to current installation location of Detectron2
# TODO create own config
# cfgFile = "/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# cfg.merge_from_file(cfgFile)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
# cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("can_val", )
# predictor = DefaultPredictor(cfg)

# make prediction
# output = predictor(im)
# print(output)
