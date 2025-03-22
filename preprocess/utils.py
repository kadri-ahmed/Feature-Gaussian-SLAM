
import torch
import torch.nn.functional as F
import os
from segment_anything import sam_model_registry
import subprocess

def load_GROUNDED_SAM_model():
    # Create the checkpoints directory
    checkpoint_dir = "/mnt/scratch/FeatureGSLAM/Grounded-Segment-Anything"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # List of URLs and corresponding file names
    checkpoints = {
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth": "sam_vit_h_4b8939.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth": "groundingdino_swint_ogc.pth",
        "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth": "ram_swin_large_14m.pth",
        "https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth": "tag2text_swin_14m.pth"
    }

    for url, filename in checkpoints.items():
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(file_path):
            print(f"{filename} already exists. Skipping download.")
        else:
            print(f"Downloading {filename} from {url} ...")
            subprocess.run(["wget", url, "-O", file_path], check=True)

    print("All checkpoints are ready.")

def load_SAM_model(checkpoint_path='SAM/checkpoints/sam_vit_b_01ec64.pth', model_type='vit_b'):
    os.makedirs("SAM/checkpoints", exist_ok=True)
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    if not os.path.exists(checkpoint_path):
        torch.hub.download_url_to_file(checkpoint_url, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Checkpoint is available at {checkpoint_path}")

    sam = None
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        print("Model loaded successfully.")
    except KeyError as e:
        print(f"Invalid model type '{model_type}'. Available: {list(sam_model_registry.keys())}")
    except Exception as e:
        print(f"Error during SAM model loading: {e}")
    return sam

def get_SAM_mask_and_feat(gt_sam_mask, level=3, filter_th=50, original_mask_feat=None, sample_mask=False):
    """
    input: 
        gt_sam_mask[4, H, W]: mask id
    output:
        mask_id[H, W]: The ID of the mask each pixel belongs to (0 indicates invalid pixels)
        mask_bool[num_mask+1, H, W]: Boolean, note that the return value excludes the 0th mask (invalid points)
        invalid_pix[H, W]: Boolean, invalid pixels
    """
    # (1) mask id: -1, 1, 2, 3,...
    mask_id = gt_sam_mask[level].clone()
    if level > 0:
        # subtract the maximum mask ID of the previous level
        mask_id = mask_id - (gt_sam_mask[level-1].max().detach().cpu()+1)
    if mask_id.min() < 0:
        mask_id = mask_id.clamp_min(-1)    # -1, 0~num_mask
    mask_id += 1    # 0, 1~num_mask+1
    invalid_pix = mask_id==0    # invalid pixels

    # (2) mask id[H, W] -> one-hot/mask_bool [num_mask+1, H, W]
    instance_num = mask_id.max()
    one_hot = F.one_hot(mask_id.type(torch.int64), num_classes=int(instance_num.item() + 1))
    # bool mask [num+1, H, W]
    mask_bool = one_hot.permute(2, 0, 1)
    
    # # TODO modify -------- only keep the largest 50
    # if instance_num > 50:
    #     top50_values, _ = torch.topk(mask_bool.sum(dim=(1,2)), 50, largest=True)
    #     filter_th = top50_values[-1].item()
    # # modify --------

    # # TODO: not used
    # # (3) delete small mask 
    # saved_idx = mask_bool.sum(dim=(1,2)) >= filter_th  # default 50 pixels
    # # Random sampling, not actually used
    # if sample_mask:
    #     prob = torch.rand(saved_idx.shape[0])
    #     sample_ind = prob > 0.5
    #     saved_idx = saved_idx & sample_ind.cuda()
    # saved_idx[0] = True  # Keep the mask for invalid points, ensuring that mask_id == 0 corresponds to invalid pixels.
    # mask_bool = mask_bool[saved_idx]    # [num_filt, H, W]

    # update mask id
    mask_id = torch.argmax(mask_bool, dim=0)  # [H, W] The ID of the pixels after filtering is 0
    invalid_pix = mask_id==0

    # TODO not used!
    # (4) Get the language features corresponding to the masks (used for 2D-3D association in the third stage)
    if original_mask_feat is not None:
        mask_feat = original_mask_feat.clone()       # [num_mask, 512]
        max_ind = int(gt_sam_mask[level].max())+1
        min_ind = int(gt_sam_mask[level-1].max())+1 if level > 0 else 0
        mask_feat = mask_feat[min_ind:max_ind, :]
        # # update mask feat
        # mask_feat = mask_feat[saved_idx[1:]]    # The 0th element of saved_idx is the mask corresponding to invalid pixels and has no features

        return mask_id, mask_bool[1:, :, :], mask_feat, invalid_pix
    return mask_id, mask_bool[1:, :, :], invalid_pix

if __name__ == "__main__":
    load_GROUNDED_SAM_model()