import os
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from open3d.pipelines import registration
import open3d as o3d
from replica_constants import REPLICA_EXISTING_CLASSES_ROOM0, REPLICA_EXISTING_CLASSES_ROOM1, REPLICA_EXISTING_CLASSES_ROOM2,REPLICA_EXISTING_CLASSES_OFFICE0, REPLICA_EXISTING_CLASSES_OFFICE1, REPLICA_EXISTING_CLASSES_OFFICE2, REPLICA_EXISTING_CLASSES_OFFICE3, REPLICA_EXISTING_CLASSES_OFFICE4,REPLICA_CLASSES, REPLICA_EXISTING_CLASSES
import json

subset_classes = REPLICA_EXISTING_CLASSES_OFFICE2

def rotate_points(points, angle_deg, axis):
    r = R.from_euler(axis, angle_deg, degrees=True)
    return r.apply(points)

def translate_points(points, translation):
    return points + np.array(translation)


def load_instance_to_class_map(json_path):
    """
    Reads Habitat's `info_semantic.json` and extracts a mapping from object ID → class ID.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a mapping {object_id: class_id}
    instance_to_class = {obj["id"]: obj["class_id"] for obj in data["objects"]}
    return instance_to_class

def create_object_to_subset_class_map(json_path, replica_existing_classes):
    """
    Creates a mapping from object ID to subset class ID.
    
    The subset class ID is determined by the position (index) of the object's
    class_id within `replica_existing_classes`. If the class_id is not in the list,
    -1 is used.
    """
    # First, get the mapping from object id to full class id.
    instance_to_class = load_instance_to_class_map(json_path)
    # Create a mapping from full class id to subset class id.
    class_to_subset = {class_id: i for i, class_id in enumerate(replica_existing_classes)}
    
    # Build a mapping: object id → subset class id.
    object_to_subset_class = {}
    for obj_id, class_id in instance_to_class.items():
        subset_class = class_to_subset.get(class_id, -1)
        object_to_subset_class[obj_id] = subset_class

    return object_to_subset_class

def read_from_ply(file_path, json_path, num_classes=101):
    """
    Reads a PLY file and extracts:
      - Vertex coordinates (points)
      - Semantic class labels (full class IDs, mapped from object_id)
      - Subset class labels (subset class IDs, according to )

    Steps:
      1. Load semantic info from `info_semantic.json`
      2. Map each object's `object_id` to `class_id`
      3. Map the full `class_id` to a subset class id (using the allowed list)
      4. Assign the labels to each vertex based on face data
    """
    # Load the two mappings:
    instance_to_class = load_instance_to_class_map(json_path)
    class_indices = []
    if num_classes == 101:
        class_indices = [i for i in range(len(REPLICA_CLASSES))]
    elif num_classes == 51:
        class_indices = REPLICA_EXISTING_CLASSES
    elif num_classes == 27: 
        class_indices = subset_classes

    object_to_subset_class = create_object_to_subset_class_map(json_path, class_indices)

    # Read the PLY file.
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data

    # Extract vertex coordinates: shape (N, 3)
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    # Initialize arrays for full class IDs and subset class IDs with -1 (unknown)
    num_points = len(points)
    class_ids = np.full(num_points, -1, dtype=np.int32)
    subset_class_ids = np.full(num_points, -1, dtype=np.int32)

    # If face data exists and includes object IDs, use it to assign vertex labels.
    if 'face' in ply_data and 'object_id' in ply_data['face'].data.dtype.names:
        face_data = ply_data['face'].data
        face_object_ids = np.array(face_data['object_id'])  # (num_faces,)
        face_vertex_indices = face_data['vertex_indices']     # List of lists

        for i, indices in enumerate(face_vertex_indices):
            object_id = face_object_ids[i]  # Get object ID for this face
            # Map to full class ID; if object_id is not found, default to -1.
            full_class = instance_to_class.get(object_id, -1)
            # Map to subset class id.
            subset_class = object_to_subset_class.get(object_id, -1)

            # Assign these labels to all vertices of the face.
            idx = np.array(indices)
            class_ids[idx] = full_class
            subset_class_ids[idx] = subset_class

    return points, class_ids, subset_class_ids


def save_colored_ply(filepath, xyz, rgb):
    if rgb.dtype != np.uint8:
        rgb_255 = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_255 = rgb
    num_verts = xyz.shape[0]
    vertex_data = np.zeros(num_verts, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    vertex_data["x"]     = xyz[:, 0]
    vertex_data["y"]     = xyz[:, 1]
    vertex_data["z"]     = xyz[:, 2]
    vertex_data["red"]   = rgb_255[:, 0]
    vertex_data["green"] = rgb_255[:, 1]
    vertex_data["blue"]  = rgb_255[:, 2]
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=True).write(filepath)
    print(f"Saved colored point cloud to: {filepath}")

def calculate_metrics(gt, pred, total_classes=20):
    """
    Expects gt, pred in [0..(total_classes)], ignoring 0 as unlabeled.
    """
    gt = gt.cpu()
    pred = pred.cpu()

    pred[gt == 0] = 0  # ignore unlabeled in prediction

    intersection = torch.zeros(total_classes)
    union        = torch.zeros(total_classes)
    correct      = torch.zeros(total_classes)
    total        = torch.zeros(total_classes)

    for cls_id in range(1, total_classes):
        intersection[cls_id] = torch.sum((gt == cls_id) & (pred == cls_id)).item()
        union[cls_id]        = torch.sum((gt == cls_id) | (pred == cls_id)).item()
        correct[cls_id]      = torch.sum((gt == cls_id) & (pred == cls_id)).item()
        total[cls_id]        = torch.sum(gt == cls_id).item()

    ious = torch.zeros(total_classes)
    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    # Classes actually appearing in GT
    gt_classes = torch.unique(gt)
    gt_classes = gt_classes[gt_classes != 0]
    mean_iou = ious[gt_classes].mean().item() if len(gt_classes) > 0 else 0.0

    valid_mask = (gt != 0)
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points  = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else 0.0

    class_accuracy = torch.zeros(total_classes)
    non_zero_mask = total != 0
    class_accuracy[non_zero_mask] = correct[non_zero_mask] / total[non_zero_mask]
    mean_class_accuracy = class_accuracy[gt_classes].mean().item() if len(gt_classes) > 0 else 0.0

    return ious, mean_iou, accuracy, mean_class_accuracy


def registration_vol_ds(source, target, init_transformation, volume, dTau, threshold, iterations):
    """
    Dummy volume-based registration (point-to-plane ICP) within a crop volume.
    """
    # Crop source and target with the provided volume
    s = source#volume.crop_point_cloud(source)
    t = target#volume.crop_point_cloud(target)
    # Downsample with voxel size dTau
    s_down = s.voxel_down_sample(dTau)
    t_down = t.voxel_down_sample(dTau)
    # Run ICP (point-to-plane)
    t_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
    )
    reg = registration.registration_icp(
        s_down, t_down, threshold, init_transformation,
        registration.TransformationEstimationPointToPlane(),
        registration.ICPConvergenceCriteria(max_iteration=iterations)
    )

    # Compose the transformation with the initial one
    reg.transformation = reg.transformation @ init_transformation
    return reg

if __name__ == "__main__":

    scene_name = "office2"
    num_classes_list = [101]
    #num_classes_list = [51, 27]
    scene_list = [scene_name]
    dataset_dir = '/mnt/datasets/Replica_v2/vmap'
    habitat = f"/mnt/projects/FeatureGSLAM/Replica_v2/vmap/room_0/habitat"
    json_path = f"{habitat}/info_semantic.json"

    for scene_name in scene_list:
        for num_classes in num_classes_list:
            # ------------------------------------------------------------------
            # 1) Load GT point cloud + labels
            # ------------------------------------------------------------------
            #gt_ply = os.path.join(dataset_dir, scene_name, f"{scene_name}_vh_clean_2.labels.ply")
            gt_ply = f"{habitat}/mesh_semantic.ply"
            gt_points, class_ids, subset_ids = read_from_ply(gt_ply, json_path, num_classes=num_classes)
        
            print(f"[{scene_name}] Loaded {gt_points.shape[0]} vertices. Max label={subset_ids.max()}")

            
            gt_labels = torch.from_numpy(subset_ids).long().cuda()

            # ------------------------------------------------------------------
            # 2) Load Gaussians, transforms, and "all_candidates"
            #    plus assigned_label_idx per cluster
            # ------------------------------------------------------------------

            file_name = f"registered_gaussians_{num_classes}"
            #file_name = f"gaussians_final_labeled_{num_classes}"
            dataset = "Replica"
            #gaussians_npz_path = os.path.join(gaussians_dir, scene_name, "gaussians_with_assigned_labels.npz")
            gaussians_npz_path = f"/mnt/projects/FeatureGSLAM/{dataset}/{scene_name}/{file_name}.npz"
            data = np.load(gaussians_npz_path, allow_pickle=True)
            slice = 32
            means3D = data['means3D']#[::slice]  # [num_gaussians, 3]
            all_candidates = data['all_candidates']#[::slice]   # shape [num_gaussians], dtype=object
            # If your file doesn't have 'all_candidates', you need a version that does.

            # assigned_label_idx => array of shape [num_clusters], each in [0..40]
            assigned_label_idx = data['assigned_label_idx']

            print("Groundtruth points extents:")
            print("min:", gt_points.min(axis=0), "max:", gt_points.max(axis=0))
            print("Means3D extents:")
            print("min:", means3D.min(axis=0), "max:", means3D.max(axis=0))

            num_gaussians = means3D.shape[0]
            if num_gaussians != len(all_candidates):
                raise ValueError(
                    f"Mismatch: means3D has {num_gaussians} gaussians, "
                    f"but all_candidates has {len(all_candidates)} entries."
                )

            # ------------------------------------------------------------------
            # 4) For each vertex, find nearest Gaussian => gather all cluster labels
            #    If GT label is in that set, we say it's correct => pred_labels[i] = GT label
            #    else pred_labels[i] = 0
            # ------------------------------------------------------------------
            nn_idx_path = "./cached_nn_idx.npz" 
            if os.path.exists(nn_idx_path):
                # Load precomputed nn_idx
                print("Loading cached nearest neighbor indices...")
                data = np.load(nn_idx_path)
                nn_idx = data["nn_idx"]
                dist = data["dist"]
            else:
                # Compute nearest neighbors using KD-Tree
                print("Computing nearest neighbors using KD-Tree...")
                kd_tree = cKDTree(means3D)
                dist, nn_idx = kd_tree.query(gt_points, k=1)

                # Save to file for future runs
                np.savez(nn_idx_path, nn_idx=nn_idx, dist=dist)
                print("Saved nearest neighbor indices for future use.")

            print("Nearest neighbor distance statistics:")
            print("Min distance:", dist.min())
            print("Max distance:", dist.max())
            print("Mean distance:", dist.mean())

            # We'll store final predicted label for each vertex
            pred_labels = np.zeros(gt_points.shape[0], dtype=np.int32)

            
            for i in range(gt_points.shape[0]):
                # The nearest gaussian index
                g = nn_idx[i]

                # Gather candidate cluster indices => e.g. [4, 7, 12]
                candidate_clusters = all_candidates[g]  # a Python list

                candidate_labels = assigned_label_idx[candidate_clusters]

                # If the GT label is unlabeled or 0 => skip
                # else check if GT label is in candidate_labels
                gt_label_i = gt_labels[i].item()

                candidate_class_names = []
                gt_class_name = []
                # **Convert class indices to class names**
                if num_classes == 101:
                    candidate_class_names = [REPLICA_CLASSES[idx] for idx in candidate_labels if idx < len(REPLICA_CLASSES)]
                    gt_class_name = REPLICA_CLASSES[gt_label_i] if gt_label_i < len(REPLICA_CLASSES) else "unknown"
                elif num_classes == 51:
                    candidate_class_names = [REPLICA_CLASSES[REPLICA_EXISTING_CLASSES[idx]] for idx in candidate_labels if idx < len(REPLICA_EXISTING_CLASSES)]
                    gt_class_name = REPLICA_CLASSES[REPLICA_EXISTING_CLASSES[gt_label_i]] if gt_label_i < len(REPLICA_EXISTING_CLASSES) else "unknown"
                elif num_classes == 27: 
                    candidate_class_names = [REPLICA_CLASSES[subset_classes[idx]] for idx in candidate_labels if idx < len(subset_classes)]
                    gt_class_name = REPLICA_CLASSES[subset_classes[gt_label_i]] if gt_label_i < len(subset_classes) else "unknown"

                
                # Print information
                if i % 5000 == 0:
                    print(f"Point {i}: GT Label = {gt_class_name}, Candidate Classes = {candidate_class_names}")
                if gt_label_i != 0:
                    if gt_label_i in candidate_labels:
                        # success => set predicted label to GT
                        pred_labels[i] = gt_label_i
                    else:
                        pred_labels[i] = 0  # mismatch => treat as unlabeled

            # Convert to torch for metrics
            pred_labels_t = torch.from_numpy(pred_labels).long().cuda()
            pred_class_names = []
            if num_classes == 101:
                pred_class_names = [REPLICA_CLASSES[idx] for idx in np.unique(pred_labels)]
            elif num_classes == 51:
                pred_class_names = [REPLICA_CLASSES[REPLICA_EXISTING_CLASSES[idx]] for idx in np.unique(pred_labels)]
            elif num_classes == 27: 
                pred_class_names = [REPLICA_CLASSES[subset_classes[idx]] for idx in np.unique(pred_labels)]

            # ------------------------------------------------------------------
            # 5) Calculate metrics (using your existing function)
            # ------------------------------------------------------------------
            max_label = max(gt_labels.max().item(), assigned_label_idx.max())
            total_classes = max_label + 1   
            _, mean_iou, accuracy, mean_class_accuracy = calculate_metrics(gt_labels, pred_labels_t, total_classes=total_classes)
            print(f"[{scene_name}] mIoU={mean_iou:.4f}, "
                f"Overall Acc={accuracy:.4f}, "
                f"MeanClassAcc={mean_class_accuracy:.4f}")

            # ------------------------------------------------------------------
            # 6) Color visualization
            #    green = correct, red = incorrect, black = unlabeled
            # ------------------------------------------------------------------
            unlabeled_mask = (gt_labels == 0)
            correct_mask   = (gt_labels == pred_labels_t) & (~unlabeled_mask)

            colors = np.zeros((gt_points.shape[0], 3), dtype=np.float32)  # default black

            # correct => green
            correct_mask_np = correct_mask.cpu()
            colors[correct_mask_np] = [0.0, 1.0, 0.0]

            # mismatch => red
            incorrect_mask = (~correct_mask) & (~unlabeled_mask)
            incorrect_mask = incorrect_mask.cpu()
            colors[incorrect_mask] = [1.0, 0.0, 0.0]

            # save colored PLY
            output_ply = f"/mnt/projects/FeatureGSLAM/{dataset}/{scene_name}/{file_name}_multi-label_prediction_visual.ply"
            save_colored_ply(output_ply, gt_points, colors)