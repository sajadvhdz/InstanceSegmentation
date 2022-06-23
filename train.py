from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

from utils import *

# %%
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output/instance_segmentation"
num_classes = 2

device = "cuda"

train_dataset_name = "dset_train"
train_images_path = "train"
train_json_annot_path = "results1.json"

test_dataset_name = "dset_test"
test_images_path = "test"
test_json_annot_path = "results2.json"

# %%

register_coco_instances(name = train_dataset_name, metadata= {},
                        json_file= train_json_annot_path, image_root= train_images_path)

register_coco_instances(name = test_dataset_name, metadata= {},
                        json_file= test_json_annot_path, image_root= test_images_path)

plot_samples(dataset_name= train_dataset_name, n=2)