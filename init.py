import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np
from detectron2.data.datasets import register_coco_instances

def get_dicts:
    return

DatasetCatalog.register("my_dataset", get_dicts)

# register_coco_instances("cans", {}, "/dataset.json", "/img")

can_metadata = MetadataCatalog.get("cans")

img = cv2.imread('A_01_03_06_2019315_175644.png')

visualizer = Visualizer(img[:, :, ::-1], metadata=can_metadata, scale=0.5)
vis = visualizer.draw_dataset_dict({
    "id": 3715,
    "dataset_id": 1,
    "category_ids": [],
    "path": "/datasets/Can-Check/A_01_03_06_2019315_175644.png",
    "width": 1100,
    "height": 657,
    "file_name": "A_01_03_06_2019315_175644.png",
    "annotated": false,
    "annotating": [],
    "num_annotations": 0,
    "metadata": {},
    "deleted": false,
    "milliseconds": 0,
    "events": [],
    "regenerate_thumbnail": false
})
cv2_imshow(vis.get_image()[:, :, ::-1])