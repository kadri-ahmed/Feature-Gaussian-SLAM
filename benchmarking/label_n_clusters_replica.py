import numpy as np
import torch
import torch.nn.functional as F
import clip
import json
from replica_constants import REPLICA_EXISTING_CLASSES_ROOM0, REPLICA_EXISTING_CLASSES_ROOM1, REPLICA_EXISTING_CLASSES_ROOM2, REPLICA_EXISTING_CLASSES_OFFICE0, REPLICA_EXISTING_CLASSES_OFFICE1, REPLICA_EXISTING_CLASSES_OFFICE2, REPLICA_EXISTING_CLASSES_OFFICE3, REPLICA_EXISTING_CLASSES_OFFICE4,REPLICA_CLASSES, REPLICA_EXISTING_CLASSES

subset_classes = REPLICA_EXISTING_CLASSES_OFFICE2

def collect_candidates_for_gaussians(kmeans_assignments):
    """
    For each Gaussian, collect the list of candidate cluster indices
    (based on kmeans_assignments) and return them as a Python list of lists.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert the kmeans_assignments to a boolean tensor
    kmeans_assignments_tensor = torch.tensor(kmeans_assignments, device=device, dtype=torch.bool)
    num_gaussians = kmeans_assignments_tensor.shape[0]

    # Prepare a list that will hold candidate indices for each Gaussian
    all_candidates_list = []

    for i in range(num_gaussians):
        # Get all clusters that are "True" for Gaussian i
        candidate_indices = torch.nonzero(kmeans_assignments_tensor[i]).squeeze(1)

        if candidate_indices.numel() == 0:
            # No candidates
            all_candidates_list.append([])
        else:
            # Store all candidates as a list of Python ints
            all_candidates_list.append(candidate_indices.cpu().tolist())

    return all_candidates_list

def label_cluster_features(cluster_lang_features, text_features, text_labels, device="cuda:0"):
    """
    Assign a single best label for each cluster by comparing
    cluster_lang_features to text_features via maximum similarity.
    Returns a 1D tensor (cluster_best_label_ids) with shape [num_clusters].
    """
    text_features = text_features.to(device)
    cluster_lang_features_tensor = cluster_lang_features.half().to(device)

    # Normalize
    text_features = F.normalize(text_features, p=2, dim=1)
    cluster_features_normalized = F.normalize(cluster_lang_features_tensor, p=2, dim=1)

    # Compute similarity [num_clusters, num_text_labels]
    similarities = cluster_features_normalized @ text_features.t()

    # For each cluster (row), pick the label with the highest similarity
    cluster_best_label_ids = torch.argmax(similarities, dim=1)
    return cluster_best_label_ids

def load_instance_to_semantic_map(json_path):
    """
    Reads Habitat's info_semantic.json and extracts the instance-to-semantic mapping.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract class mapping
    class_mapping = {cls["id"]: cls["name"] for cls in data["classes"]}

    # Extract instance-to-class mapping
    instance_to_class = data["id_to_label"]

    # Convert instance IDs to semantic labels
    instance_to_semantic = {
        instance_id: class_mapping.get(class_id, "unknown")
        for instance_id, class_id in enumerate(instance_to_class)
    }

    return instance_to_semantic

def load_text_embeddings(num_classes=101, device="cuda:0"):
    if num_classes == 101:
        class_names = REPLICA_CLASSES
    elif num_classes == 51:
        class_names = [REPLICA_CLASSES[i] for i in REPLICA_EXISTING_CLASSES]
    elif num_classes == 27: 
        class_names = [REPLICA_CLASSES[i] for i in subset_classes]

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_prompts = class_names
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
    return class_names, text_features

if __name__ == "__main__":
    dataset = "Replica"
    scene_name = "office2"
    #scene_name = "office3"
    file_name = "gaussians_final"
    num_classes = 101
    # Example paths
    #gaussians_npz = "/mnt/projects/FeatureGSLAM/scannet/scene0000_00/gaussians.npz"
    #gaussians_npz = f"/mnt/projects/FeatureGSLAM/{dataset}/{scene_name}/{file_name}.npz"
    gaussians_npz = f"/mnt/projects/FeatureGSLAM/benchmarking_vitus/{scene_name}_69_mapping_quality_0/{file_name}.npz"
    out_path = f"/mnt/projects/FeatureGSLAM/{dataset}/{scene_name}/{file_name}_labeled_{num_classes}.npz"
    # Load your data
    data = np.load(gaussians_npz)
    means3D = data["means3D"]
    kmeans_assignments = data["kmeans_assignments"]
    cluster_lang_features = data["cluster_lang_features"]  # shape [num_clusters, clip_dim]
    text_labels, text_features = load_text_embeddings(num_classes=num_classes)
    for i, label in enumerate(text_labels):
        print(f"{i}: {label}")
    # -----------------------------------------------------
    # Fix mismatch by clamping means3D and kmeans_assignments to same length
    # -----------------------------------------------------
    num_gaussians_m3d = means3D.shape[0]
    num_gaussians_kma = kmeans_assignments.shape[0]
    if num_gaussians_m3d != num_gaussians_kma:
        # Choose the smaller of the two
        num_gaussians = min(num_gaussians_m3d, num_gaussians_kma)
        print(f"Warning: Mismatch in shapes. means3D has {num_gaussians_m3d}, "
              f"kmeans_assignments has {num_gaussians_kma}. "
              f"Clamping both to {num_gaussians}.")
        means3D = means3D[:num_gaussians]
        kmeans_assignments = kmeans_assignments[:num_gaussians]

    # 2) Assign a label to each cluster (via CLIP)
    cluster_lang_features_tensor = torch.from_numpy(cluster_lang_features)
    assigned_label_idx = label_cluster_features(
        cluster_lang_features_tensor,
        text_features,
        text_labels
    ).cpu().numpy()


    # 1) Gather all candidate cluster indices for each Gaussian
    all_candidates_list = collect_candidates_for_gaussians(kmeans_assignments)

    # 3) Convert list of lists to an object array (ragged data) for saving
    all_candidates_obj = np.array(all_candidates_list, dtype=object)

    # 4) Save results
    np.savez(
        out_path,
        means3D=means3D,                   # [num_gaussians, 3]
        all_candidates=all_candidates_obj,  # List of candidate cluster indices per Gaussian
        cluster_lang_features=cluster_lang_features,  # [num_clusters, clip_dim]
        assigned_label_idx=assigned_label_idx         # [num_clusters] -> best text label ID
    )

    print(f"Saved to {out_path}")