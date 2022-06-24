import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = 'IS_cfg.pickle'

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

import time
import glob, os
os.chdir("stierman/")
for file in glob.glob("*.jpg"):
    print(file)
    image_path = file

onImage(image_path, predictor)