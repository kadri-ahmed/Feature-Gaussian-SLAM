import numpy as np
import os
import torch

def save_gaussians_with_lang_by_cluster(
    gaussians_npz_path,
    params,
    kmeans,
    cluster_features,  # Tensor of shape [num_clusters, 512] containing language features
    device='cuda'
):
    means3D = params['means3D'].detach().cpu().numpy()
    means3D[:, 2] = -means3D[:, 2]  
    means3D[:, 1] = -means3D[:, 1] 

   # Keep the PyTorch version for indexing
    cluster_lang_features_tensor = torch.stack(
        [cluster_features[idx] for idx in sorted(cluster_features.keys())]
    ).detach().cpu()

    # Convert to NumPy for saving
    cluster_lang_features = cluster_lang_features_tensor.numpy()  # Shape [num_clusters, 512]

    codebook = kmeans.assignments.cpu().numpy()
    os.makedirs(os.path.dirname(gaussians_npz_path), exist_ok=True)
    np.savez(
        gaussians_npz_path,
        means3D=means3D,
        rgb_colors=params['rgb_colors'].detach().cpu().numpy(),
        unnorm_rotations=params['unnorm_rotations'].detach().cpu().numpy(),
        logit_opacities=params['logit_opacities'].detach().cpu().numpy(),
        log_scales=params['log_scales'].detach().cpu().numpy(),
        cluster_lang_features=cluster_lang_features,  # [num_clusters, 512]
        kmeans_assignments=codebook  # [num_gaussians, num_clusters]
    )


def save_gaussians_with_lang_by_cluster_single_dim_codebook(
    gaussians_npz_path,
    params,
    kmeans,
    cluster_features,  # Tensor of shape [num_clusters, 512] containing language features
    device='cuda'
):

    # Compute minimum cluster index per Gaussian, ignoring -1
    cluster_indices = torch.amax(kmeans.assignments, dim=1)
    num_clusters = cluster_indices.max() + 1  # Number of clusters
    filtered_gaussians = []
    filtered_lang_features = []

    for cluster_idx in range(num_clusters):
        if torch.all(cluster_features[cluster_idx] == 0):
            print(f"Skipping cluster {cluster_idx} due to missing language feature.")
            continue

        filtered_gaussians_idx = torch.nonzero(cluster_indices == cluster_idx, as_tuple=True)[0]
        if filtered_gaussians_idx.numel() == 0:
            print(f"No Gaussians found for cluster {cluster_idx}.")
            continue

        lang_feature = cluster_features[cluster_idx].detach().cpu().numpy()

        means3D = params['means3D'][filtered_gaussians_idx].detach().cpu().numpy()
        means3D[:, 2] = -means3D[:, 2]  
        means3D[:, 1] = -means3D[:, 1]  
        filtered_gaussians.append({
            'means3D': means3D,
            'rgb_colors': params['rgb_colors'][filtered_gaussians_idx, 3:6].detach().cpu().numpy(),
            'unnorm_rotations': params['unnorm_rotations'][filtered_gaussians_idx].detach().cpu().numpy(),
            'logit_opacities': params['logit_opacities'][filtered_gaussians_idx].detach().cpu().numpy(),
            'log_scales': params['log_scales'][filtered_gaussians_idx].detach().cpu().numpy()
        })
        filtered_lang_features.append(np.repeat(lang_feature[None, :], filtered_gaussians_idx.size(0), axis=0))

    combined_gaussians = {key: np.concatenate([g[key] for g in filtered_gaussians], axis=0) for key in filtered_gaussians[0]}
    combined_lang_features = np.concatenate(filtered_lang_features, axis=0)

    os.makedirs(os.path.dirname(gaussians_npz_path), exist_ok=True)
    np.savez(
        gaussians_npz_path,
        means3D=combined_gaussians['means3D'],
        rgb_colors=combined_gaussians['rgb_colors'],
        unnorm_rotations=combined_gaussians['unnorm_rotations'],
        logit_opacities=combined_gaussians['logit_opacities'],
        log_scales=combined_gaussians['log_scales'],
        gaussian_lang_feat=combined_lang_features
    )
    print(f"Processed clusters: {len(filtered_gaussians)}")
    print(f"Total Gaussians saved: {combined_gaussians['means3D'].shape[0]}")

    #print_gaussians_with_language_features(gaussians_npz_path)

    print(f"Saved Gaussians with language features to: {gaussians_npz_path}")



def print_gaussians_with_language_features(npz_path, num_to_print=5):
    """
    Print a few Gaussians along with their associated language features.
    
    Args:
        npz_path (str): Path to the NPZ file with Gaussian data.
        num_to_print (int): Number of Gaussians to print.
    """
    data = np.load(npz_path)
    
    means3D = data['means3D']  # [num_gaussians, 3]
    lang_features = data['gaussian_lang_feat']  # [num_gaussians, 512]

    print(f"Number of Gaussians: {means3D.shape[0]}")
    print(f"Language feature dimension: {lang_features.shape[1]}")
    print(f"\nPrinting the first {num_to_print} Gaussians and their language features:\n")
    
    for i in range(min(num_to_print, means3D.shape[0])):
        print(f"Gaussian {i + 1}:")
        print(f"  3D Position: {means3D[i]}")
        print(f"  Language Feature (first 5 dimensions): {lang_features[i][:5]}")
        print(f"  Language Feature Norm: {np.linalg.norm(lang_features[i]):.4f}")
        print("-" * 50)
