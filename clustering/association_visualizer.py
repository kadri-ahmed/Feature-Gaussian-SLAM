import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from benchmarking.label_gaussians_scannet import load_text_embeddings


def plot_single_cluster(rendered_features,
                        image,
                        out,
                        out_file_prefix="visualization/cluster_visualization"):
    cmap = plt.get_cmap("tab20")
    C, H, W = rendered_features.shape
    
    individual_image = torch.zeros((3, H, W), dtype=torch.float32)

    cluster_feat = rendered_features  # [C, H, W]
    nonzero_mask = (cluster_feat.abs().sum(dim=0) > 0)
    color = cmap(0)[:3]
    color_tensor = torch.tensor(color, dtype=torch.float32)
    individual_image[:, nonzero_mask] = color_tensor.view(3, 1)

    # Overlay both images onto the original image
    base_img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def overlay_clusters(accum_img, base, out_filename):
        accum_np = accum_img.numpy().transpose(1, 2, 0)  # [H, W, 3]
        accum_np_255 = (accum_np * 255).astype(np.uint8)
        blended = cv2.addWeighted(base, 1 - 0.7, accum_np_255, 0.7, 0)
        Image.fromarray(blended).save(out_filename)
#        print(f"Saved: {out_filename}")
        return blended

    # Output paths
    os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
    individual_out_file = f"{out_file_prefix}_{out}_individual.png"

    # overlay_clusters(grouped_image, base_img, grouped_out_file)
    overlay_clusters(individual_image, base_img, individual_out_file)
    

def plot_cluster_language_association(
    rendered_cluster_features,
    per_cluster_lang_feat,  # CLIP features [num_clusters, embedding_dim]
    image,                  # Real image (C, H, W)
    cluster_indices,
    out_file_prefix="visualization/cluster_visualization",
    alpha=0.7,
    similarity_threshold=0.93
):
    num_clusters = len(rendered_cluster_features)
    if num_clusters == 0:
        print("No clusters to visualize!")
        return
    cmap = plt.get_cmap("tab20")

    C, H, W = rendered_cluster_features[0].shape
#    
    individual_image = torch.zeros((3, H, W), dtype=torch.float32)

    for cluster_idx in cluster_indices:
        cluster_feat = rendered_cluster_features[cluster_idx]  # [C, H, W]
        nonzero_mask = (cluster_feat.abs().sum(dim=0) > 0)
        color = cmap(cluster_idx % 20)[:3]
        color_tensor = torch.tensor(color, dtype=torch.float32)
        individual_image[:, nonzero_mask] = color_tensor.view(3, 1)

    # Overlay both images onto the original image
    base_img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def overlay_clusters(accum_img, base, out_filename):
        accum_np = accum_img.numpy().transpose(1, 2, 0)  # [H, W, 3]
        accum_np_255 = (accum_np * 255).astype(np.uint8)
        blended = cv2.addWeighted(base, 1 - alpha, accum_np_255, alpha, 0)
        Image.fromarray(blended).save(out_filename)
#        print(f"Saved: {out_filename}")
        return blended

    # Output paths
    os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
    grouped_out_file = f"{out_file_prefix}_grouped.png"
    individual_out_file = f"{out_file_prefix}_individual.png"

    # overlay_clusters(grouped_image, base_img, grouped_out_file)
    overlay_clusters(individual_image, base_img, individual_out_file)


def plot_mask_cluster_iou(
    image,
    rendered_cluster_features,
    sam_masks,
    association_results,
    output_dir="visualization",
    filename="cluster_iou_overlay.png",
    alpha_union=0.5,  # Transparency for the union of cluster and SAM masks
    alpha_sam=0.3,    # Transparency for the SAM mask silhouette
    alpha_cluster=0.2 # Transparency for the cluster mask silhouette
):
    os.makedirs(output_dir, exist_ok=True)

    # Convert image to NumPy array in [0, 255] range
    overlay_img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Aggregate cluster features across channels if necessary
    clusters_list = [rendered_cluster_features[key] for key in rendered_cluster_features]
    cluster_features = torch.stack(clusters_list)  # Shape [num_clusters, C, H, W]
    cluster_masks = cluster_features.sum(dim=1) > 0  # [num_clusters, H, W]

    num_clusters, H, W = cluster_masks.shape
    num_masks = sam_masks.shape[0]

    # Convert to NumPy
    cluster_masks = cluster_masks.cpu().numpy()
    sam_masks_np = sam_masks.cpu().numpy()
    assoc = association_results.cpu().numpy()  # Shape [num_clusters, num_masks, 2]

    # Random colors for clusters
    colors = np.random.rand(num_clusters, 3)

    # Initialize overlay
    vis_overlay = overlay_img.copy()

    # First, modify the pixels on vis_overlay according to all mask operations
    for c_idx in range(num_clusters):
        cluster_mask = cluster_masks[c_idx]  # Shape [H, W]
        color_c = colors[c_idx]

        for m_idx in range(min(num_masks, 50)):
            score = assoc[c_idx, m_idx]
            matched_bool = assoc[c_idx, m_idx]

            if score != 0:
                sam_mask = sam_masks_np[m_idx]  # Shape [H, W]

                # Union of cluster and SAM masks
                union_mask = cluster_mask | sam_mask

                # Overlay: Blend union mask with the image
                vis_overlay[union_mask, 0] = (
                    alpha_union * vis_overlay[union_mask, 0] 
                    + (1 - alpha_union) * color_c[0] * 255
                )
                vis_overlay[union_mask, 1] = (
                    alpha_union * vis_overlay[union_mask, 1] 
                    + (1 - alpha_union) * color_c[1] * 255
                )
                vis_overlay[union_mask, 2] = (
                    alpha_union * vis_overlay[union_mask, 2] 
                    + (1 - alpha_union) * color_c[2] * 255
                )

                # Add SAM mask silhouette
                vis_overlay[sam_mask, 0] = (
                    alpha_sam * vis_overlay[sam_mask, 0] + (1 - alpha_sam) * 100
                )
                vis_overlay[sam_mask, 1] = (
                    alpha_sam * vis_overlay[sam_mask, 1] + (1 - alpha_sam) * 100
                )
                vis_overlay[sam_mask, 2] = (
                    alpha_sam * vis_overlay[sam_mask, 2] + (1 - alpha_sam) * 100
                )

                # Add cluster mask silhouette
                vis_overlay[cluster_mask, 0] = (
                    alpha_cluster * vis_overlay[cluster_mask, 0] + (1 - alpha_cluster) * 50
                )
                vis_overlay[cluster_mask, 1] = (
                    alpha_cluster * vis_overlay[cluster_mask, 1] + (1 - alpha_cluster) * 50
                )
                vis_overlay[cluster_mask, 2] = (
                    alpha_cluster * vis_overlay[cluster_mask, 2] + (1 - alpha_cluster) * 200
                )

    # Now create a figure and axes to draw the image + text
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(vis_overlay)
    ax.axis("off")
    ax.set_title(f"Clusters + SAM Masks (matches) - total clusters={num_clusters}, total masks={num_masks}")

    # Once the image is drawn, we place the text on top.
    # We iterate again to find centroids and place text:
    for c_idx in range(num_clusters):
        cluster_mask = cluster_masks[c_idx]
        color_c = colors[c_idx]

        for m_idx in range(min(num_masks, 50)):
            score = assoc[c_idx, m_idx]
            matched_bool = assoc[c_idx, m_idx]

            if score != 0:
                sam_mask = sam_masks_np[m_idx]
                union_mask = cluster_mask | sam_mask

                ys, xs = np.where(union_mask)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())  # Centroid
                    ax.text(
                        cx, cy, f"{score:.2f}",
                        color="black",
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor=color_c, alpha=1.0, 
                            edgecolor="black", boxstyle="round"
                        )
                    )

    # Save the visualization
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)
    print(f"Saved overlay to {out_path}")



def plot_mask_cluster_iou_with_labels(
    image,
    current_time_idx,
    rendered_cluster_features,
    sam_masks,
    association_results_clusterwise,  # dict { cluster_idx -> list of (time_idx, mask_indices, scores) }
    cluster_lang_feat,                # [num_clusters, 512]
    output_dir="visualization",
    filename="cluster_iou_with_clip_labels.png",
    alpha_union=0.5,    # Transparency for union overlay
    alpha_sam=0.3,      # Transparency for SAM silhouette
    alpha_cluster=0.2,  # Transparency for cluster silhouette
    score_threshold=0.5 # Score above which we consider a match
):
    """
    Overlays matched cluster masks and SAM masks on `image`, annotates each matched 
    region with the cluster’s best text label + similarity. Accepts a *clusterwise*
    association result rather than a fixed 3D tensor.

    Args:
        image: (C,H,W) float Tensor in [0,1] range
        current_time_idx: integer, the time index you want to visualize
        rendered_cluster_features: dict {cluster_idx -> (C,H,W) or (H,W) tensor} 
            Non-zero indicates the cluster region (like a silhouette).
        sam_masks: (num_masks, H, W) BoolTensor for the *current_time_idx*
        association_results_clusterwise: dict
            {cluster_idx -> [(frame_idx, mask_indices, scores), ...]}
        cluster_lang_feat: (num_clusters, 512) float Tensor (e.g., CLIP embeddings)
        output_dir: str, directory to save output
        filename: str, output filename
        alpha_union: float, union overlay transparency
        alpha_sam: float, SAM silhouette overlay transparency
        alpha_cluster: float, cluster silhouette overlay transparency
        score_threshold: float, minimum score to consider a cluster-mask match
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert image from (C,H,W) [0,1] to uint8 [0,255]
    overlay_img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Build cluster_masks as a NumPy array [num_clusters, H, W]
    cluster_indices = sorted(rendered_cluster_features.keys())
    num_clusters = len(cluster_indices)

    cluster_masks_list = []
    for c_idx in cluster_indices:
        feat = rendered_cluster_features[c_idx]  # shape: (C,H,W) or (H,W)
        if feat.dim() == 3:
            # sum across channel dimension => (H,W) boolean
            cmask = (feat.sum(dim=0) > 0).cpu().numpy()
        else:
            # feat is already (H,W)
            cmask = (feat > 0).cpu().numpy()
        cluster_masks_list.append(cmask)

    cluster_masks = np.stack(cluster_masks_list, axis=0)  # [num_clusters, H, W]
    sam_masks_np = sam_masks.cpu().numpy()                # [num_masks, H, W]

    # Prepare random colors for clusters
    colors = np.random.rand(num_clusters, 3)

    # Copy overlay for blending
    vis_overlay = overlay_img.copy()

    # Loop over each cluster
    for idx_in_list, c_idx in enumerate(cluster_indices):
        # c_idx is the actual cluster ID, idx_in_list is 0..num_clusters-1
        cluster_mask = cluster_masks[idx_in_list]  # shape (H,W)
        color_c = colors[idx_in_list]

        # Filter association data for this cluster at the current_time_idx
        # association_results_clusterwise[c_idx] = 
        #   [(time_idx, mask_indices, scores), ...]
        cluster_assoc_list = association_results_clusterwise.get(c_idx, [])
        # keep only those with frame_idx == current_time_idx
        cluster_assoc_list = [
            (f_idx, m_inds, sc) 
            for (f_idx, m_inds, sc) in cluster_assoc_list
            if f_idx == current_time_idx
        ]

        # If there's no association info for this cluster/time, skip
        if not cluster_assoc_list:
            continue

        # We might have multiple sets of (mask_indices, scores) for the same frame
        # if the code appended them multiple times. We'll combine them all:
        combined_mask_indices = []
        combined_scores = []
        for (_, m_inds, sc) in cluster_assoc_list:
            combined_mask_indices.append(m_inds)
            combined_scores.append(sc)

        # Merge them into one tensor
        combined_mask_indices = torch.cat(combined_mask_indices) if combined_mask_indices else torch.tensor([], dtype=torch.long)
        combined_scores = torch.cat(combined_scores) if combined_scores else torch.tensor([])

        # For each mask in combined_mask_indices
        for i in range(len(combined_mask_indices)):
            m_idx = combined_mask_indices[i].item()
            score = combined_scores[i].item()

            if m_idx < 0 or m_idx >= sam_masks_np.shape[0]:
                # skip invalid mask index
                continue
            if score < score_threshold:
                # skip low-scoring matches
                continue

            sam_mask = sam_masks_np[m_idx]
            union_mask = cluster_mask | sam_mask  # boolean union

            # ----------- Blend the union mask -----------
            vis_overlay[union_mask, 0] = (
                alpha_union * vis_overlay[union_mask, 0]
                + (1 - alpha_union) * color_c[0] * 255
            )
            vis_overlay[union_mask, 1] = (
                alpha_union * vis_overlay[union_mask, 1]
                + (1 - alpha_union) * color_c[1] * 255
            )
            vis_overlay[union_mask, 2] = (
                alpha_union * vis_overlay[union_mask, 2]
                + (1 - alpha_union) * color_c[2] * 255
            )

            # SAM silhouette
            vis_overlay[sam_mask, 0] = (
                alpha_sam * vis_overlay[sam_mask, 0] + (1 - alpha_sam) * 100
            )
            vis_overlay[sam_mask, 1] = (
                alpha_sam * vis_overlay[sam_mask, 1] + (1 - alpha_sam) * 100
            )
            vis_overlay[sam_mask, 2] = (
                alpha_sam * vis_overlay[sam_mask, 2] + (1 - alpha_sam) * 100
            )

            # Cluster silhouette
            vis_overlay[cluster_mask, 0] = (
                alpha_cluster * vis_overlay[cluster_mask, 0] + (1 - alpha_cluster) * 50
            )
            vis_overlay[cluster_mask, 1] = (
                alpha_cluster * vis_overlay[cluster_mask, 1] + (1 - alpha_cluster) * 50
            )
            vis_overlay[cluster_mask, 2] = (
                alpha_cluster * vis_overlay[cluster_mask, 2] + (1 - alpha_cluster) * 200
            )

    # ----- Display / annotate text labels -----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(vis_overlay)
    ax.axis("off")
    ax.set_title(f"Clusters + SAM Masks (matches)\nnum_clusters={num_clusters}, num_masks={sam_masks_np.shape[0]}")

    # Suppose you have some text embeddings / labels to get the best label:
    text_labels, text_features = load_text_embeddings()  # user-defined
    # text_features: [num_text_labels, 512], normalized
    # cluster_lang_feat: [num_clusters, 512]
    device_for_text = text_features.device
    cluster_lang_feat = cluster_lang_feat.to(device_for_text)

    # We'll annotate each cluster's centroid with its top label
    for idx_in_list, c_idx in enumerate(cluster_indices):
        # L2 normalize the cluster embedding if needed
        feat = cluster_lang_feat[c_idx].float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

        # Cosine similarity
        sim_scores = feat @ text_features.float().T  # shape [num_text_labels]
        top_label_id = sim_scores.argmax().item()
        best_label = text_labels[top_label_id]
        best_score = sim_scores[top_label_id].item()

        # Approx centroid of the cluster mask
        cluster_mask = cluster_masks[idx_in_list]
        ys, xs = np.where(cluster_mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())  # centroid
            annotation_text = f"{best_label} ({best_score:.2f})"
            color_c = colors[idx_in_list]

            ax.text(
                cx, cy,
                annotation_text,
                color="black",
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=color_c, alpha=1.0,
                    edgecolor="black", boxstyle="round"
                )
            )

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)
    print(f"Saved overlay to {out_path}")
# def plot_mask_cluster_iou_with_labels(
#     image,
#     rendered_cluster_features,
#     sam_masks,
#     association_results,     # we still need this to check if matched_bool > 0.5
#     cluster_lang_feat,       # shape [num_clusters, 512]
#     output_dir="visualization",
#     filename="cluster_iou_with_clip_labels.png",
#     alpha_union=0.5,   # Transparency for the union of cluster and SAM masks
#     alpha_sam=0.3,     # Transparency for the SAM mask silhouette
#     alpha_cluster=0.2  # Transparency for the cluster mask silhouette
# ):
#     """
#     This function overlays matched cluster masks and SAM masks on `image` 
#     and annotates each matched region with:
#         - The best matching text label (via CLIP)
#         - The CLIP similarity score for that label

#     Arguments:
#         image: (C, H, W) torch.Tensor
#         rendered_cluster_features: dict {key -> torch.Tensor of shape (C, H, W)} 
#         sam_masks: (num_masks, H, W) torch.BoolTensor
#         association_results: (num_clusters, num_masks, 2) torch.Tensor 
#             [:, :, 0] = "score" (IoU-like) – (not displayed here)
#             [:, :, 1] = "matched_bool" (1 or 0) – used to determine if we annotate 
#         cluster_lang_feat: (num_clusters, 512) torch.Tensor 
#             CLIP-like embeddings for each cluster (unnormalized or normalized)
#         output_dir: str, output directory to save the figure
#         filename: str, output filename
#         alpha_union: float, transparency for union overlay
#         alpha_sam: float, transparency for SAM silhouettes
#         alpha_cluster: float, transparency for cluster silhouettes
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Convert image from (C,H,W) and [0,1] range to uint8 [0,255].
#     overlay_img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

#     # Aggregate cluster features across channels to get binary masks
#     clusters_list = [rendered_cluster_features[k] for k in rendered_cluster_features]
#     cluster_features = torch.stack(clusters_list)  # [num_clusters, C, H, W]
#     cluster_masks = cluster_features.sum(dim=1) > 0  # [num_clusters, H, W]

#     num_clusters, H, W = cluster_masks.shape
#     num_masks = sam_masks.shape[0]

#     # Convert masks and associations to NumPy
#     cluster_masks = cluster_masks.cpu().numpy()
#     sam_masks_np = sam_masks.cpu().numpy()
#     assoc = association_results.cpu().numpy()  # [num_clusters, num_masks, 2]

#     # Random colors for each cluster
#     colors = np.random.rand(num_clusters, 3)

#     # --------- First pass: blend overlays -------------
#     vis_overlay = overlay_img.copy()
#     for c_idx in range(num_clusters):
#         cluster_mask = cluster_masks[c_idx]
#         color_c = colors[c_idx]

#         for m_idx in range(min(num_masks, 50)):
#             matched_bool = assoc[c_idx, m_idx, 1]  # either 0 or 1
#             if matched_bool > 0.5:
#                 sam_mask = sam_masks_np[m_idx]
#                 union_mask = cluster_mask | sam_mask

#                 # Blend the union mask with overlay
#                 vis_overlay[union_mask, 0] = (
#                     alpha_union * vis_overlay[union_mask, 0] 
#                     + (1 - alpha_union) * color_c[0] * 255
#                 )
#                 vis_overlay[union_mask, 1] = (
#                     alpha_union * vis_overlay[union_mask, 1] 
#                     + (1 - alpha_union) * color_c[1] * 255
#                 )
#                 vis_overlay[union_mask, 2] = (
#                     alpha_union * vis_overlay[union_mask, 2] 
#                     + (1 - alpha_union) * color_c[2] * 255
#                 )

#                 # Add SAM mask silhouette
#                 vis_overlay[sam_mask, 0] = (
#                     alpha_sam * vis_overlay[sam_mask, 0] + (1 - alpha_sam) * 100
#                 )
#                 vis_overlay[sam_mask, 1] = (
#                     alpha_sam * vis_overlay[sam_mask, 1] + (1 - alpha_sam) * 100
#                 )
#                 vis_overlay[sam_mask, 2] = (
#                     alpha_sam * vis_overlay[sam_mask, 2] + (1 - alpha_sam) * 100
#                 )

#                 # Add cluster mask silhouette
#                 vis_overlay[cluster_mask, 0] = (
#                     alpha_cluster * vis_overlay[cluster_mask, 0] 
#                     + (1 - alpha_cluster) * 50
#                 )
#                 vis_overlay[cluster_mask, 1] = (
#                     alpha_cluster * vis_overlay[cluster_mask, 1] 
#                     + (1 - alpha_cluster) * 50
#                 )
#                 vis_overlay[cluster_mask, 2] = (
#                     alpha_cluster * vis_overlay[cluster_mask, 2] 
#                     + (1 - alpha_cluster) * 200
#                 )

#     # --------- Prepare figure -------------
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(vis_overlay)
#     ax.axis("off")
#     ax.set_title(
#         f"Clusters + SAM Masks (matches)\n"
#         f"total clusters={num_clusters}, total masks={num_masks}"
#     )

#     # --------- Second pass: compute & display CLIP label + similarity -------------
#     # Load CLIP text embeddings
#     text_labels, text_features = load_text_embeddings()  
#     # text_features is shape [num_text_labels, 512], normalized
#     # cluster_lang_feat is shape [num_clusters, 512]

#     # Ensure they are on the same device
#     device_for_text = text_features.device  # e.g., 'cuda:0'
#     cluster_lang_feat = cluster_lang_feat.to(device_for_text)

#     # For each cluster, compute the best label
#     for c_idx in range(num_clusters):
#         # L2 normalize if not already normalized
#         feat = cluster_lang_feat[c_idx]
#         feat = feat.to(text_features.device)
#         feat = feat.to(text_features.dtype)
#         feat = feat / feat.norm(dim=-1, keepdim=True)

#         # Cosine similarity with text_features
#         sim_scores = feat @ text_features.T  # shape: [num_text_labels]
#         top_label_id = sim_scores.argmax().item()
#         best_label = text_labels[top_label_id]
#         best_score = sim_scores[top_label_id].item()  # float

#         cluster_mask = cluster_masks[c_idx]
#         color_c = colors[c_idx]

#         for m_idx in range(min(num_masks, 50)):
#             matched_bool = assoc[c_idx, m_idx, 1]
#             if matched_bool > 0.5:
#                 sam_mask = sam_masks_np[m_idx]
#                 union_mask = cluster_mask | sam_mask

#                 ys, xs = np.where(union_mask)
#                 if len(xs) > 0:
#                     cx, cy = int(xs.mean()), int(ys.mean())  # centroid
#                     annotation_text = f"{best_label} ({best_score:.2f})"
#                     ax.text(
#                         cx, cy,
#                         annotation_text,
#                         color="black",
#                         fontsize=10,
#                         ha="center",
#                         va="center",
#                         bbox=dict(
#                             facecolor=color_c, alpha=1.0, 
#                             edgecolor="black", boxstyle="round"
#                         )
#                     )

#     # --------- Save & clean up -------------
#     out_path = os.path.join(output_dir, filename)
#     fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
#     plt.close(fig)
#     print(f"Saved overlay to {out_path}")