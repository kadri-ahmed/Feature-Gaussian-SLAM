import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import argparse

import torch
import numpy as np
import torchvision

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from importlib.machinery import SourceFileLoader

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
    ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset,
    TUMDataset, ScannetPPDataset, NeRFCaptureDataset
)

_BASE_DIR = f"{_BASE_DIR}/Grounded-Segment-Anything"
sys.path.insert(0, _BASE_DIR)

from automatic_label_ram_demo import load_model, get_grounding_output, show_mask, show_box, load_image
import GroundingDINO.groundingdino.datasets.transforms as T

from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

from segment_anything import build_sam, SamPredictor

import cv2
import torch.nn.functional as F
import clip

def resize_bboxes(bboxes, original_size, target_size):
    orig_width, orig_height = original_size
    target_width, target_height = target_size

    # Compute scaling factors.
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    # Resize each coordinate.
    resized_bboxes = bboxes.copy()
    resized_bboxes[:, 0] *= scale_x  # x1
    resized_bboxes[:, 1] *= scale_y  # y1
    resized_bboxes[:, 2] *= scale_x  # x2
    resized_bboxes[:, 3] *= scale_y  # y2

    return resized_bboxes

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
    if masks.size == 0:
        return masks
    return F.interpolate(masks.float(), size=(height, width), mode='nearest').squeeze(1).bool()

def labels_to_clip(labels):
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=clip_device)
    text_inputs = clip.tokenize(labels).to(clip_device)
    with torch.no_grad():
        clip_features = clip_model.encode_text(text_inputs).cpu().numpy()
    return clip_features

def get_file_path(base, time_idx):
    if "scannet" in base:
        return f"{base}/frames/color/{time_idx}.jpg"
    elif "Replica" in base:
        return f"{base}/results/frame{time_idx:06d}.jpg"
    
def process_frames_distributed(
    device_ids,
    frame_indices,
    dataset,
    output_path,
    config_file,
    ram_ckpt,
    gdino_ckpt,
    sam_ckpt,
    box_threshold,
    text_threshold,
    iou_threshold,
    dataset_cfg,
    input_folder
):
    gdino_sam_device = torch.device(f"cuda:{device_ids[0]}")
    ram_device = torch.device(f"cuda:{device_ids[1]}")

    print(f"Processing frames with RAM on GPU {device_ids[0]} and GroundingDINO/SAM on GPU {device_ids[1]}...")

    ram_model = ram(pretrained=ram_ckpt, image_size=384, vit="swin_l").to(ram_device).eval()

    groundingdino_model = load_model(config_file, gdino_ckpt, device=gdino_sam_device)

    sam_predictor = SamPredictor(build_sam(checkpoint=sam_ckpt).to(gdino_sam_device))

    os.makedirs(f"{output_path}/sam_grounded_masks", exist_ok=True)

    desired_height, desired_width = dataset_cfg["desired_image_height"], dataset_cfg["desired_image_width"]
    paths = dataset.get_filepaths()
    #for time_idx in tqdm(frame_indices, desc=f"Processing frames"):
    for time_idx, path in enumerate(paths[0]):
        output_file = Path(output_path) / f"sam_grounded_masks/{time_idx}_groundedsam.npz"
        if output_file.exists():
            continue

        #image_path = get_file_path(input_folder, time_idx)
        image_path = path
        image_pil, image = load_image(image_path)

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            normalize,
        ])
        raw_image = transform(image_pil.resize((384, 384))).unsqueeze(0).to(ram_device)
        tags, _ = inference_ram(raw_image, ram_model)

        boxes_filt, scores, pred_phrases = get_grounding_output(
            groundingdino_model,
            image.to(gdino_sam_device),
            tags,
            box_threshold,
            text_threshold,
            device=gdino_sam_device,
        )

        boxes_filt = boxes_filt.cpu()
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        if boxes_filt.shape[0] > 0:
            keep_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[keep_idx]
            pred_phrases = [pred_phrases[idx] for idx in keep_idx]
        else:
            pred_phrases = []

        image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if len(boxes_filt) > 0:
            sam_predictor.set_image(image_np)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2]).to(
                gdino_sam_device
            )
            masks, sam_scores, _ = sam_predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
            )
        else:
            masks = np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool)

        original_size = (W, H)
        target_size = (desired_width, desired_height)
        np.savez_compressed(
            output_file,
            masks=resize_sam_masks(masks, desired_height, desired_width).cpu().numpy() if masks.size != 0 else np.array([]),
            mask_scores=sam_scores.cpu().numpy() if masks.size != 0 else np.array([]),
            boxes=resize_bboxes(boxes_filt.cpu().numpy(), original_size, target_size) if masks.size != 0 else np.array([]),
            labels=np.array(pred_phrases) if masks.size != 0 else np.array([]),
            tags=tags if masks.size != 0 else np.array([]),
            clip=labels_to_clip(pred_phrases) if masks.size != 0 else np.array([])
        )

        if time_idx % 5 == 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(image_np)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)

            output_dir = f"{input_folder}/grounded_plots"
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

    config_file = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

    ram_checkpoint = "Grounded-Segment-Anything/ram_swin_large_14m.pth"

    sam_checkpoint = "Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
    sam_hq_checkpoint = "Grounded-Segment-Anything/sam_hq_vit_h.pth"
    use_sam_hq = False

    device = "cuda"

    output_folder = "sam_grounded_masks"

    box_threshold = 0.25
    text_threshold = 0.2
    iou_threshold = 0.5

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

    output_prefix = dataset.input_folder.split('/')
    # If path was something like "dir1/dir2/file", then parts = ["dir1", "dir2", "file"]
    # Remove the first element (dir1) and rejoin the rest
    output_prefix = '/'.join(output_prefix[2:])
    output_path = f"/mnt/projects/FeatureGSLAM/{output_prefix}"
    #output_path = os.path.join(dataset.input_folder, output_folder)
    os.makedirs(output_path, exist_ok=True)

    num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0
    print(f"Number of available CUDA GPUs = {num_gpus}")

    if num_gpus > 1:
        process_frames_distributed([0,1], [i for i in range(num_frames)], dataset, output_path,
                config_file,
                ram_checkpoint,
                grounded_checkpoint,
                sam_checkpoint,
                box_threshold,
                text_threshold,
                iou_threshold,
                dataset_config,
                output_path
        )
    else:
        print("Use 2 GPUs!")

if __name__ == "__main__":
    main()
