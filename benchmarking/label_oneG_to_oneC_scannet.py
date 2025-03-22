import numpy as np
import torch
import os
import clip
import torch.nn.functional as F  

def assign_labels_to_gaussians(kmeans_assignments):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kmeans_assignments_tensor = torch.tensor(kmeans_assignments, device=device, dtype=torch.bool)
    cluster_lang_features_tensor = torch.tensor(cluster_lang_features, device=device, dtype=torch.float)
    num_gaussians = kmeans_assignments_tensor.shape[0]
    assigned_clusters = torch.empty(num_gaussians, device=device, dtype=torch.int64)
    for i in range(num_gaussians):
        candidate_indices = torch.nonzero(kmeans_assignments_tensor[i]).squeeze(1)
        if candidate_indices.numel() == 0:
            assigned_clusters[i] = -1
            continue
        if candidate_indices.numel() == 1:
            assigned_clusters[i] = candidate_indices[0]
            continue
        candidate_features = cluster_lang_features_tensor[candidate_indices]
        candidate_features_normalized = F.normalize(candidate_features, p=2, dim=1)
        sim_matrix = candidate_features_normalized @ candidate_features_normalized.t()
        distances = 1 - sim_matrix
        total_distances = distances.sum(dim=1)
        best_candidate_index = torch.argmin(total_distances)
        best_cluster = candidate_indices[best_candidate_index]
        assigned_clusters[i] = best_cluster
    assigned_clusters_cpu = assigned_clusters.cpu().numpy()   
    return assigned_clusters_cpu

def load_text_embeddings(device="cuda:0"):
    text_labels = {
        0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
        6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
        11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
        16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
        21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
        26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
        31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
        36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
    }
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_prompts = [f"A photo of a {label}" for label in text_labels.values()]
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
    return text_labels, text_features

def label_cluster_features(cluster_lang_features, text_features, text_labels, device="cuda:0"):
    text_features = text_features.to(device)
    cluster_lang_features_tensor = cluster_lang_features.half().to(device)
    text_features = F.normalize(text_features, p=2, dim=1)
    cluster_features_normalized = F.normalize(cluster_lang_features_tensor, p=2, dim=1)
    similarities = cluster_features_normalized @ text_features.t()
    cluster_best_label_ids = torch.argmax(similarities, dim=1)
    return cluster_best_label_ids

if __name__ == "__main__":
    gaussians_npz = "/mnt/projects/FeatureGSLAM/gaussians_best.npz"
    out_path = "FeatureGSLAM/benchmarking/gaussians_with_assigned_labels.npz"
    data = np.load(gaussians_npz)
    means3D = data['means3D']
    kmeans_assignments = data['kmeans_assignments']
    cluster_lang_features = torch.from_numpy(data['cluster_lang_features'])
    text_labels, text_features = load_text_embeddings()
    cluster_features_to_label_id = label_cluster_features(cluster_lang_features, text_features, text_labels)
    gaussians_to_cluster = assign_labels_to_gaussians(kmeans_assignments)
    np.savez(out_path,
             means3D=means3D,
             oneD_assignments=gaussians_to_cluster,
             cluster_lang_features=cluster_lang_features.cpu().numpy(),
             assigned_label_idx=cluster_features_to_label_id.cpu().numpy())
