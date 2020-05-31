import flask
from flask_cors import CORS
from flask import request, jsonify
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import cv2
import json
import requests
import numpy as np

def score_image(predictor: DefaultPredictor):
    # load an image of Lionel Messi with a ball
    im = cv2.imread('/test/A_01_03_06_2019315_175348.png')

    # make prediction
    return predictor(im)

def get_dicts():
    with open('dataset.json') as json_file:
        data = json.load(json_file)
    return data

def prepare_pridctor():
    DatasetCatalog.register("can_train", get_dicts)
    MetadataCatalog.get("can_train").set(thing_classes=["Sticker", "Edding", "Verschmutzung"])
    can_metadata = MetadataCatalog.get("can_train")
    dataset_dicts = get_dicts()

    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfgFile = "/usr/local/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    return (predictor, classes)

app = flask.Flask(__name__)
CORS(app)
predictor, classes = prepare_pridctor()

@app.route("/api/score-image", methods=["POST"])
def process_score_image_request():
    scoring_result = score_image(predictor)

    instances = scoring_result["instances"]
    scores = instances.get_fields()["scores"].tolist()
    pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
        "classes": classes
    }

    return jsonify(response)

app.run(host="0.0.0.0", port=5000)