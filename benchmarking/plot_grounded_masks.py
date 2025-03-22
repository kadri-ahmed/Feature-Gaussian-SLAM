import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import argparse

import torch
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from importlib.machinery import SourceFileLoader

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
    ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset,
    TUMDataset, ScannetPPDataset, NeRFCaptureDataset
)


import cv2
import torch.nn.functional as F

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")
    
def resize_sam_masks(masks, height, width):
    masks = masks.unsqueeze(0)
    resized_masks = F.interpolate(masks.float(), size=(height, width), mode='nearest')
    # Remove unnecessary dimensions and return as boolean mask
    return resized_masks.squeeze(0).squeeze(0).bool()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label, mask_id):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, f'{label}   {mask_id}')

def get_file_path(base, time_idx):
    if "scannet" in base:
        return f"{base}/frames/color/{time_idx}.jpg"
    elif "Replica" in base:
        return f"{base}/results/frame{time_idx:06d}.jpg"

def process_frames(
    device_id,
    frame_indices,
    output_path,
    input_folder
):
    device = torch.device(f"cuda:{device_id}")

    print(f"Processing frames on GPU {device_id}...")

    for time_idx in tqdm(frame_indices, desc=f"Processing on GPU {device_id}"):
        #input_path = Path(output_path) / f"{time_idx}_groundedsam.npz"
        input_path = f"/mnt/projects/FeatureGSLAM/Replica/room0/sam_grounded_masks/{time_idx}_groundedsam.npz"
        npz_data = np.load(input_path)
        masks = torch.from_numpy(npz_data['masks'])
        masks = resize_sam_masks(masks, 680, 1200)
        boxes_filt = torch.from_numpy(npz_data['boxes'])
        pred_phrases = npz_data['labels'].tolist()
        image_path = get_file_path(input_folder, time_idx)

        image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        for mask_id in range(masks.size(0)):
            show_mask(masks[mask_id].cpu().numpy(), plt.gca(), random_color=True)
        for i, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
            show_box(box.numpy(), plt.gca(), label, i)

#        output_dir = f"{input_folder}/grounded_plots"
        output_dir = "/mnt/projects/FeatureGSLAM/Replica/room0/grounded_plots"

        os.makedirs(output_dir, exist_ok=True)
        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, f"{time_idx}_plot.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()  

    device = "cuda"

    output_folder = "sam_grounded_masks"

    config = experiment.config
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {"dataset_name": dataset_config["dataset_name"]}
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    print("Loading dataset ...")
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=torch.device(device),
        relative_pose=True,
        ignore_bad=dataset_config.get("ignore_bad", False),
        use_train_split=dataset_config.get("use_train_split", True),
    )

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
    print(f"Dataset loaded with {len(dataset)} total frames. Processing up to {num_frames} frames.")

    output_path = os.path.join(dataset.input_folder, output_folder)
    os.makedirs(output_path, exist_ok=True)

    num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0
    print(f"Number of available CUDA GPUs = {num_gpus}")



    frame_indices = list(range(num_frames))
    process_frames(
        0,
        frame_indices,
        output_path,
        dataset.input_folder
    )
 
if __name__ == "__main__":
    main()
