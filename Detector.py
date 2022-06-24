from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
from matplotlib import pyplot as plt
import numpy as np


class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # Load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.SEGMENTATION)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        plt.imshow(output.get_image()[:, :, ::-1])
        plt.show()

    def onVideo(self):
        cap = cv2.VideoCapture(0)

        # cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened()==False):
            print("Error opening the file...")
            return

        (success, image)= cap.read()

        while success:
            predictions = self.predictor(image)

            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                             instance_mode=ColorMode.SEGMENTATION)
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            plt.imshow(output.get_image()[:, :, ::-1])
            plt.show()
            plt.waitforbuttonpress(.001)
            (success, image) = cap.read()
