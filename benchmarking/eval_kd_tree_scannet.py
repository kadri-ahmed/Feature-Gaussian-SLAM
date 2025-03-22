import os
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

nyu40_dict = {
    0:  "unlabeled",  1:  "wall", 2:  "floor", 3:  "cabinet", 4:  "bed",
    5:  "chair",      6:  "sofa", 7:  "table", 8:  "door",    9:  "window",
    10: "bookshelf",  11: "picture", 12: "counter", 13: "blinds", 14: "desk",
    15: "shelves",    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror",
    20: "floormat",   21: "clothes", 22: "ceiling", 23: "books",  24: "refrigerator",
    25: "television", 26: "paper",  27: "towel",   28: "showercurtain", 29: "box",
    30: "whiteboard", 31: "person", 32: "nightstand", 33: "toilet", 34: "sink",
    35: "lamp",       36: "bathtub", 37: "bag",    38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

def rotate_points(points, angle_deg, axis):
    r = R.from_euler(axis, angle_deg, degrees=True)
    return r.apply(points)

def translate_points(points, translation):
    return points + np.array(translation)

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    # Coordinates: (x, y, z)
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    labels = vertex_data['label']  
    return points, labels


def save_colored_ply(filepath, xyz, rgb):
    """
    Save points with colors to a PLY file using plyfile.
    :param filepath: Output path, e.g. 'result.ply'
    :param xyz: (N,3) NumPy array of point coordinates
    :param rgb: (N,3) NumPy array of RGB in [0..1], or [0..255]
    """
    # Ensure rgb is in 0..255 range (uint8) for standard PLY viewers
    if rgb.dtype != np.uint8:
        rgb_255 = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_255 = rgb

    num_verts = xyz.shape[0]

    # Create a structured array suitable for plyfile
    vertex_data = np.zeros(num_verts, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    vertex_data["x"]    = xyz[:, 0]
    vertex_data["y"]    = xyz[:, 1]
    vertex_data["z"]    = xyz[:, 2]
    vertex_data["red"]  = rgb_255[:, 0]
    vertex_data["green"]= rgb_255[:, 1]
    vertex_data["blue"] = rgb_255[:, 2]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=True).write(filepath)
    print(f"Saved colored point cloud to: {filepath}")


def calculate_metrics(gt, pred, total_classes=20):
    gt = gt.cpu()
    pred = pred.cpu()

    pred[gt == 0] = 0  # ignore unlabeled

    intersection = torch.zeros(total_classes)
    union        = torch.zeros(total_classes)
    correct      = torch.zeros(total_classes)
    total        = torch.zeros(total_classes)

    for cls_id in range(1, total_classes):
        intersection[cls_id] = torch.sum((gt == cls_id) & (pred == cls_id)).item()
        union[cls_id]        = torch.sum((gt == cls_id) | (pred == cls_id)).item()
        correct[cls_id]      = torch.sum((gt == cls_id) & (pred == cls_id)).item()
        total[cls_id]        = torch.sum(gt == cls_id).item()

    # IoU
    ious = torch.zeros(total_classes)
    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    # Which classes appear in this scene?
    gt_classes = torch.unique(gt)
    gt_classes = gt_classes[gt_classes != 0]  # ignore unlabeled
    mean_iou = ious[gt_classes].mean().item() if len(gt_classes) > 0 else 0.0

    # Overall accuracy
    valid_mask = gt != 0
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points  = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else 0.0

    # Per-class accuracy => mAcc
    class_accuracy = torch.zeros(total_classes)
    non_zero_mask = total != 0
    class_accuracy[non_zero_mask] = correct[non_zero_mask] / total[non_zero_mask]
    mean_class_accuracy = class_accuracy[gt_classes].mean().item() if len(gt_classes) > 0 else 0.0

    return ious, mean_iou, accuracy, mean_class_accuracy

if __name__ == "__main__":
    target_id = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36
    ]
    target_id_mapping = {}
    for new_idx, old_label in enumerate(target_id, start=1):
        target_id_mapping[old_label] = new_idx

    scene_list = ['scene0000_00']
    dataset_dir = './data/scannet'
    gaussians_dir = './benchmarking'

    for scene_name in scene_list:
        gt_ply = os.path.join(dataset_dir, scene_name, f"{scene_name}_vh_clean_2.labels.ply")
        points, labels = read_labels_from_ply(gt_ply)
        print(f"[{scene_name}] Loaded {points.shape[0]} vertices. Max label={labels.max()}")

        # Remap ground truth labels to 1..19 if they exist in target_id
        new_labels = np.zeros_like(labels, dtype=np.int32)
        for old_value, new_value in target_id_mapping.items():
            mask = (labels == old_value)
            new_labels[mask] = new_value

        gt_labels = torch.from_numpy(new_labels).long().cuda()

        gaussians_npz_path = os.path.join(gaussians_dir, scene_name, "gaussians_with_assigned_labels.npz")
        data = np.load(gaussians_npz_path)
        means3D  = data['means3D']  # shape [num_gaussians, 3]
        # Flip axes as needed
        means3D[:, 2] = -means3D[:, 2]  
        means3D[:, 1] = -means3D[:, 1]

        # Apply your scene-specific transformation
        means3D = rotate_points(means3D, 250, 'x')
        means3D = rotate_points(means3D, 0, 'y')
        means3D = rotate_points(means3D, 159, 'z')
        means3D = translate_points(means3D, [2.76, 2.98, 1.3])

        cluster_to_label = data['assigned_label_idx']
        gaussian_cluster_assignments = data['oneD_assignments']
        gaussian_labels = cluster_to_label[gaussian_cluster_assignments]

        print("Groundtruth points extents:")
        print("min:", points.min(axis=0), "max:", points.max(axis=0))
        print("Means3D extents:")
        print("min:", means3D.min(axis=0), "max:", means3D.max(axis=0))

        # Remap predicted labels
        if len(gaussian_labels.shape) > 1:
            gaussian_labels = gaussian_labels.squeeze(-1)

        new_gaussian_labels = np.zeros_like(gaussian_labels, dtype=np.int32)
        for old_value, new_value in target_id_mapping.items():
            new_gaussian_labels[gaussian_labels == old_value] = new_value

        # Use nearest-neighbor to assign each vertex in GT to a predicted label
        kd_tree = cKDTree(means3D)
        dist, nn_idx = kd_tree.query(points, k=1)
        print("Nearest neighbor distance statistics:")
        print("Min distance:", dist.min())
        print("Max distance:", dist.max())
        print("Mean distance:", dist.mean())

        pred_labels = new_gaussian_labels[nn_idx]  # shape [num_vertices]
        pred_labels_t = torch.from_numpy(pred_labels).long().cuda()

        # Calculate metrics
        _, mean_iou, accuracy, mean_class_accuracy = calculate_metrics(gt_labels, pred_labels_t, total_classes=20)
        print(f"[{scene_name}] mIoU={mean_iou:.4f}, "
              f"Overall Acc={accuracy:.4f}, "
              f"MeanClassAcc={mean_class_accuracy:.4f}")

        ###
        unlabeled_mask = (new_labels == 0)
        correct_mask   = (new_labels == pred_labels) & ~unlabeled_mask

        # 2) Create an array of colors, one for each vertex
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)  # default is black (0,0,0)

        # 3) Color correct predictions (green)
        colors[correct_mask] = [0.0, 1.0, 0.0]   # green

        # 4) Color all other labeled points that are incorrect (red)
        incorrect_mask = (~correct_mask) & (~unlabeled_mask)
        colors[incorrect_mask] = [1.0, 0.0, 0.0]  # red
        
        # 3) Save the point cloud to a PLY file
        output_ply = f"benchmarking/{scene_name}_prediction_visual.ply"
        save_colored_ply(output_ply, points, colors)
