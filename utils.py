from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode

import random
import cv2
from matplotlib import pyplot as plt

#%%
def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        strx = s["file_name"]
        s["file_name"] = strx.replace('images\\7/', '')
        img = cv2.imread(s["file_name"])

        v = Visualizer(img[:,:,::-1], metadata= dataset_custom_metadata)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize= (15, 20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 11
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg

def onImage(imagePath, predictor):
    im = cv2.imread(imagePath)
    outputs = predictor(im)

    viz = Visualizer(im[:, :, ::-1], metadata={}, scale= 0.8,
                     instance_mode=ColorMode.IMAGE_BW)
    v = viz.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(15,15))
    plt.imshow(v.get_image()[:, :, ::-1])
    filename = imagePath.replace('.jpg', '.png')
    str_save = "predictions/pred_" + filename
    plt.savefig(str_save)
    plt.show()

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

def get_sample_inputs(sample_image = None):

    if sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs