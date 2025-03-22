import numpy as np
import open3d as o3d
import torch
import clip
from matplotlib import colormaps

# Path to your Gaussian data file
gaussians_npz_path = "benchmarking/scene0000_00/gaussians_replica.npz"

# Load Gaussian data
data = np.load(gaussians_npz_path)
means3D = data['means3D']  # shape: [num_gaussians, 3]
kmeans_assignments = data['kmeans_assignments']  # shape: [num_gaussians]
cluster_lang_features = data['cluster_lang_features']  # shape: [num_clusters, clip_dim]
color = data['rgb_colors']

clip_device = "cuda" if torch.cuda.is_available() else "cpu"

# Convert cluster CLIP features to tensor and normalize
cluster_lang_features = torch.tensor(cluster_lang_features, dtype=torch.float32).to(clip_device)
cluster_lang_features = torch.nn.functional.normalize(cluster_lang_features, dim=-1)

# Create an Open3D point cloud for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(means3D)

# Load CLIP model
print(f"Loading CLIP model on device: {clip_device}")
clip_model, preprocess = clip.load("ViT-B/32", device=clip_device)
clip_model.eval()

SIMILARITY_THRESHOLD = 0.8

while True:
    prompt_text = input("Type a query ('q' to quit): ")
    if prompt_text.lower() == "q":
        break

    # Encode user query via CLIP
    text_tokens = clip.tokenize([prompt_text]).to(clip_device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens)
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

    # Compute cosine similarity between query and cluster CLIP features
    similarity = torch.nn.functional.cosine_similarity(
        cluster_lang_features, text_feat.unsqueeze(0), dim=-1
    )
    # Similarity shape: [num_clusters]

    # Find clusters exceeding threshold
    matching_clusters = (similarity > SIMILARITY_THRESHOLD).nonzero(as_tuple=True)[1]
    print(f"Matching Clusters: {matching_clusters.cpu().numpy()}")

    if matching_clusters.numel() == 0:
        print("No clusters found above threshold. Try a different query.")
        continue

    # Iterate over columns, each containing exactly one unique cluster ID (besides -1)
    for i in range(kmeans_assignments.shape[1]):
        column = kmeans_assignments[:, i]  # shape: [num_gaussians], only one valid ID or all -1

        # Convert to torch on the same device
        column_tensor = torch.tensor(column, dtype=torch.long, device=clip_device)

        # Mask out invalid entries
        valid_mask = (column_tensor != -1)
        if not valid_mask.any():
            # This column has no valid cluster IDs
            continue

        # Find the single unique ID in this column
        unique_id_torch = torch.unique(column_tensor[valid_mask])
        if unique_id_torch.numel() == 0:
            # No valid ID
            continue

        # Convert that single ID tensor to a Python int
        cluster_id = unique_id_torch.item()

        # Check if that cluster ID is above threshold
        if similarity[:,cluster_id] > SIMILARITY_THRESHOLD:
            # Gather the corresponding Gaussians
            column_mask = (column_tensor == cluster_id)
            column_mask_cpu = column_mask.cpu().numpy()

            matched_means3D = means3D[column_mask_cpu]
            matched_colors = color[column_mask_cpu]

            if len(matched_means3D) == 0:
                continue

            print(f"Cluster {cluster_id} in column {i} has {len(matched_means3D)} Gaussians.")

            # Create Open3D geometries
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(matched_means3D)
            pcd.colors = o3d.utility.Vector3dVector(matched_colors)

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

            # Visualize
            o3d.visualization.draw_geometries([pcd, axes])