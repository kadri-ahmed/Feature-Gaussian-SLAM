import numpy as np
import torch
import os
import clip

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

def assign_labels_to_gaussians(gaussians_npz, text_labels, text_features, out_path):
    data = np.load(gaussians_npz)
    means3D = data['means3D']
    gaussian_lang_feat = data['gaussian_lang_feat']
    gaussian_lang_feat_t = torch.from_numpy(gaussian_lang_feat).float().cuda()
    text_features_t = text_features.transpose(0, 1).contiguous().float()
    sim_matrix = torch.matmul(gaussian_lang_feat_t, text_features_t)
    assigned_label_idx = torch.argmax(sim_matrix, dim=1)
    similarity_scores = sim_matrix[torch.arange(sim_matrix.size(0)), assigned_label_idx]
    assigned_label_idx_np = assigned_label_idx.cpu().numpy().astype(np.int32)
    similarity_scores_np = similarity_scores.cpu().numpy().astype(np.float32)
    assigned_label_names = [text_labels[idx] for idx in assigned_label_idx_np]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        means3D=means3D,
        assigned_label_idx=assigned_label_idx_np,
        assigned_label_names=assigned_label_names,
        similarity_scores=similarity_scores_np
    )
    print(f"Assigned labels and similarity scores saved to: {out_path}")


def view_best_gaussian_per_label(npz_path):
    data = np.load(npz_path)
    means3D = data['means3D']  # shape [num_gaussians, 3]
    assigned_label_names = data['assigned_label_names']  # shape [num_gaussians]
    similarity_scores = data['similarity_scores']  # shape [num_gaussians]

    unique_labels = np.unique(assigned_label_names)
    
    best_indices = []
    for label in unique_labels:
        label_mask = (assigned_label_names == label)
        
        label_indices = np.where(label_mask)[0]
        if len(label_indices) == 0:
            continue  
        
        label_scores = similarity_scores[label_indices]
        best_idx_in_label = label_indices[np.argmax(label_scores)]
        best_indices.append(best_idx_in_label)
    
    best_indices_sorted = sorted(
        best_indices,
        key=lambda idx: similarity_scores[idx],
        reverse=True
    )

    print(f"Best Gaussian for each label (total distinct labels = {len(unique_labels)}):")
    print(f"{'Label':<20} {'Similarity':<12} {'3D Position':<30}")
    print("-" * 70)
    for idx in best_indices_sorted:
        label = assigned_label_names[idx]
        sim   = similarity_scores[idx]
        pos   = means3D[idx]
        print(f"{label:<20} {sim:<12.4f} {pos}")



if __name__ == "__main__":
    gaussians_npz = "benchmarking/scene0000_00/gaussians.npz"
    out_path = "benchmarking/scene0000_00/gaussians_with_assigned_labels.npz"
    
    text_labels, text_features = load_text_embeddings()
    assign_labels_to_gaussians(gaussians_npz, text_labels, text_features, out_path)
    view_best_gaussian_per_label(out_path)