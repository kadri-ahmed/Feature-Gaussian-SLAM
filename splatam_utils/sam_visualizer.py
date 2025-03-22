import matplotlib.pyplot as plt
import os
import numpy as np

# Convert SAM masks to an overlay visualization and save to disk
def save_sam_output_visualization(image, sam_masks_current_frame, output_dir="sam_visualizations", filename="sam_overlay.png"):
    """
    Saves a visualization of the SAM output by overlaying masks on the original image.

    Args:
        image (numpy array): The original input image in HWC format.
        sam_masks_current_frame (list): The SAM-generated masks, each with 'segmentation' as key.
        output_dir (str): Directory to save the visualizations.
        filename (str): Filename for the saved visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a blank canvas for overlay
    overlay = image.copy()
    alpha = 0.5  # Transparency factor for overlay

    # Create a random color for each mask
    num_masks = len(sam_masks_current_frame)
    colors = np.random.randint(0, 255, (num_masks, 3), dtype=np.uint8)

    # Overlay masks on the image
    for idx, mask_data in enumerate(sam_masks_current_frame):
        mask = mask_data#['segmentation'].astype(bool)
        color = colors[idx]
        overlay[mask] = alpha * overlay[mask] + (1 - alpha) * color

    # Save the image with overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Number of Masks: {num_masks}")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    #print(f"Saved overlay visualization to {output_path}")

# Save individual masks
def save_individual_masks(image, sam_masks_current_frame, output_dir="sam_masks"):
    """
    Saves each SAM mask as an individual visualization.

    Args:
        image (numpy array): The original input image in HWC format.
        sam_masks_current_frame (list): The SAM-generated masks, each with 'segmentation' as key.
        output_dir (str): Directory to save the mask visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, mask_data in enumerate(sam_masks_current_frame):
        mask = mask_data['segmentation']
        plt.figure(figsize=(6, 6))
        plt.imshow(image, alpha=0.7)
        plt.imshow(mask, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.title(f"Mask {idx}")
        mask_filename = os.path.join(output_dir, f"mask_{idx}.png")
        plt.savefig(mask_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved individual mask visualization to {mask_filename}")

