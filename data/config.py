import os
import numpy as np
from utils import StepLR
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)

config = dict()

"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(root_path, "raw_data/train")
config["val_split"] = os.path.join(root_path, "raw_data/val")

# Preprocessed Dataset
config["preprocess"] = True  # whether use preprocess or not
config["preprocess_train"] = os.path.join(root_path, "preprocess/train_pfp.p")
config["preprocess_val"] = os.path.join(root_path, "preprocess/val_pfp.p")

config["batch_size"] = 1
config["val_batch_size"] = 1

config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6

config["val_workers"] = 0
config["workers"] = 0
config['cross_dist'] = 6
config['cross_angle'] = 0.5 * np.pi
