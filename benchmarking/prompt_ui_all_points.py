import numpy as np
import matplotlib
import open3d as o3d
import torch
import clip
from matplotlib import colormaps
# Path to your Gaussian data file
gaussians_npz_path = "benchmarking/scene0000_00/gaussians.npz"

# Load Gaussian data
data = np.load(gaussians_npz_path)
means3D = data['means3D']  # shape: [num_gaussians, 3]
clip_features = data['gaussian_lang_feat']  # shape: [num_gaussians, clip_dim]
clip_device = "cuda" if torch.cuda.is_available() else "cpu"

# Normalize clip features for cosine similarity
clip_features = torch.tensor(clip_features, dtype=torch.float32).to(clip_device)
clip_features = torch.nn.functional.normalize(clip_features, dim=-1)

# Create an Open3D point cloud for visualization
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(means3D)

# Load CLIP model
print(f"Loading CLIP model on device: {clip_device}")
clip_model, preprocess = clip.load("ViT-B/32", device=clip_device)
clip_model.eval()

# User loop for querying
while True:
    prompt_text = input("Type a query ('q' to quit): ")
    if prompt_text.lower() == "q":
        break

    # Encode user query
    text_tokens = clip.tokenize([prompt_text]).to(clip_device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens)
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(clip_features, text_feat.unsqueeze(0), dim=-1)

    # Normalize similarity scores for visualization
    similarity = (similarity + 1.0) / 2.0  # Scale to [0, 1]
    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-12)

    # Ensure similarity is a 1D array
    similarity = similarity.flatten()

    # Map similarity to colors using the updated Matplotlib colormap API
    from matplotlib import colormaps
    cmap = colormaps["jet"]  # Replaces get_cmap
    similarity_colors = cmap(similarity.cpu().numpy())[:, :3]  # Extract only RGB (discard alpha)

    # Ensure similarity_colors has the correct format
    similarity_colors = np.squeeze(similarity_colors)  # Remove unnecessary dimensions if any
    assert similarity_colors.ndim == 2, "Color data must be a 2D array."
    assert similarity_colors.shape[1] == 3, "Color data must have 3 channels (RGB)."

    # Convert to float32 and clip range to [0, 1]
    similarity_colors = similarity_colors.astype(np.float32)
    similarity_colors = np.clip(similarity_colors, 0.0, 1.0)

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(similarity_colors)
    # Visualize in Open3D
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axes])