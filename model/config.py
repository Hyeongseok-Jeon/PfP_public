from utils import StepLR
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = 'PfP_for_LaneGCN'

### config used ###
config = dict()
"""Train"""
config["display_iters"] = 205942
config["val_iters"] = 205942 * 2
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 50
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])
config["save_dir"] = os.path.join(root_path, "results", model_name)
config["test_dir"] = os.path.join(root_path, "tests", model_name)

config["batch_size"] = 2
config["val_batch_size"] = 2
config["workers"] = 0
config["val_workers"] = config["workers"]

"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(root_path, "data/raw_data/train")
config["val_split"] = os.path.join(root_path, "dataset/raw_data/val")

# Preprocessed Dataset
config["preprocess"] = True  # whether use preprocess or not
config["preprocess_train"] = os.path.join(root_path, "data", "preprocess", "train_pfp_dist_filtering.p")
config["preprocess_val"] = os.path.join(root_path, "data", "preprocess", "val_pfp_dist_filtering.p")

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 20
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["repres_coef"] = 1.0
config["reg_coef"] = 1.0
config["reconst_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2


### end of config ###
