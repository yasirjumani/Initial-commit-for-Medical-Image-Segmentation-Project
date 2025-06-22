"""
Script to calculate quantitative metrics for medical image segmentation model performance.
Requires predicted masks (from analyze_predictions.py) and ground truth masks for the test set.
"""

import os
import sys
from pathlib import Path
import shutil

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is not installed. Please install it using 'pip install Pillow'")
    sys.exit(1)

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=C0301,C0303,C0304,R0914,E0401,E1101,W0718,W0611,W0404,R0915 # Added R0915 for main function's original state

# --- Configuration ---
output_dir = Path('output')
predictions_path = output_dir / 'test_predictions.pth'

test_image_dir = Path('/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/'
                      'ISIC2018_Task1-2_Test_Input')
test_mask_dir = Path('/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/'
                     'ISIC2018_Task1_Test_GroundTruth')


# --- Data Loading and Custom Dataset for Evaluation ---
class TestEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset for loading test images and their ground truth masks for evaluation.
    Assumes image and mask filenames match except for a suffix for masks (e.g., '_segmentation').
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png')) and not f.startswith('.')
        ])

        self.mask_paths = {}
        for f in os.listdir(mask_dir):
            if f.lower().endswith(('.jpg', '.png')) and not f.startswith('.'):
                base_name = f.replace('_segmentation', '').replace('.png', '').replace('.jpg', '')
                self.mask_paths[base_name] = os.path.join(mask_dir, f)

        self.aligned_pairs = []
        for img_path in self.image_paths:
            img_base_name = Path(img_path).stem
            if img_base_name in self.mask_paths:
                self.aligned_pairs.append((img_path, self.mask_paths[img_base_name]))

        if not self.aligned_pairs:
            raise ValueError(
                f"No aligned image-mask pairs found. "
                f"Image files in {image_dir} vs. mask files in {mask_dir}. "
                "Check naming conventions or paths."
            )
        print(f"Successfully aligned {len(self.aligned_pairs)} image-mask pairs for evaluation.")

    def __len__(self):
        return len(self.aligned_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.aligned_pairs[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()

        return img, mask, Path(img_path).name, img_path


# --- Metric Functions ---
def dice_score(pred_mask, gt_mask):
    """
    Calculates the Dice Similarity Coefficient.
    Args:
        pred_mask (np.array): Predicted binary mask.
        gt_mask (np.array): Ground truth binary mask.
    Returns:
        float: Dice score.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Predicted and ground truth masks must have the same shape.")

    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0 else 0.0

    return (2.0 * intersection) / union

def iou_score(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) score.
    Args:
        pred_mask (np.array): Predicted binary mask.
        gt_mask (np.array): Ground truth binary mask.
    Returns:
        float: IoU score.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Predicted and ground truth masks must have the same shape.")

    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection

    if union == 0:
        return 1.0 if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0 else 0.0

    return intersection / union

def calculate_uncertainty_map(pred1_prob, pred2_prob):
    """
    Calculates an uncertainty map based on the absolute difference between two predictions.
    Args:
        pred1_prob (np.array): Probability map from decoder 1 (0-1 range).
        pred2_prob (np.array): Probability map from decoder 2 (0-1 range).
    Returns:
        np.array: Uncertainty map (absolute difference).
    """
    return np.abs(pred1_prob - pred2_prob)

# pylint: disable=R0913, R0917
def visualize_segmentation(original_img_path, gt_mask_np, pred_mask_np, uncertainty_map_np, dice, iou):
    """
    Displays the original image, ground truth mask, predicted mask, and uncertainty map.
    Args:
        original_img_path (str): Path to the original image file.
        gt_mask_np (np.array): Ground truth binary mask (NumPy array).
        pred_mask_np (np.array): Predicted binary mask (NumPy array).
        uncertainty_map_np (np.array): Uncertainty map (NumPy array).
        dice (float): Dice score for the current sample.
        iou (float): IoU score for the current sample.
    """
    original_image = Image.open(original_img_path).convert('RGB')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Sample: {Path(original_img_path).name} | Dice: {dice:.4f} | IoU: {iou:.4f}")

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(gt_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    axes[3].imshow(uncertainty_map_np, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title('Uncertainty Map')
    axes[3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f"visualizations/{Path(original_img_path).stem}_segmentation_and_uncertainty.png")
    plt.close(fig)

# --- Refactored Functions for Main Logic ---

def setup_visualizations_directory():
    """Removes and recreates the visualizations directory."""
    visualizations_dir = Path("visualizations")
    if visualizations_dir.exists() and visualizations_dir.is_dir():
        print(f"Removing existing visualizations directory: {visualizations_dir}")
        shutil.rmtree(visualizations_dir)
    os.makedirs(visualizations_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {visualizations_dir}")

def process_test_samples(evaluation_dataset, prediction1_tensor, prediction2_tensor, num_viz_samples):
    """
    Processes test samples, calculates metrics, and prepares data for visualization.
    Returns (all_dice_scores, all_iou_scores, samples_for_viz).
    """
    all_dice_scores = []
    all_iou_scores = []
    samples_for_viz = []

    num_test_samples = min(len(evaluation_dataset), prediction1_tensor.shape[0])

    print("\nCalculating metrics...")
    for i in range(num_test_samples):
        _, gt_mask_tensor, _, original_img_path = evaluation_dataset[i]

        pred1_prob_tensor = prediction1_tensor[i].squeeze(0)
        pred2_prob_tensor = prediction2_tensor[i].squeeze(0)

        avg_pred_prob_np = ((pred1_prob_tensor + pred2_prob_tensor) / 2.0).cpu().numpy()
        pred_mask_np = (avg_pred_prob_np > 0.5).astype(np.float32)

        gt_mask_np = gt_mask_tensor.squeeze(0).cpu().numpy()

        uncertainty_map_np = calculate_uncertainty_map(pred1_prob_tensor.cpu().numpy(), pred2_prob_tensor.cpu().numpy())

        dice = dice_score(pred_mask_np, gt_mask_np)
        iou = iou_score(pred_mask_np, gt_mask_np)

        all_dice_scores.append(dice)
        all_iou_scores.append(iou)

        if len(samples_for_viz) < num_viz_samples:
            samples_for_viz.append({
                'original_img_path': original_img_path,
                'gt_mask_np': gt_mask_np,
                'pred_mask_np': pred_mask_np,
                'uncertainty_map_np': uncertainty_map_np,
                'dice': dice,
                'iou': iou
            })

        if (i + 1) % 100 == 0 or (i + 1) == num_test_samples:
            print(f"Processed {i+1}/{num_test_samples} samples.")

    return all_dice_scores, all_iou_scores, samples_for_viz

def print_summary_results(dice_scores, iou_scores):
    """Prints the average Dice and IoU scores."""
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print("\n--- Evaluation Results ---")
    print(f"Total samples evaluated: {len(dice_scores)}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print("--------------------------")

def generate_visualizations(samples_for_viz):
    """Generates and saves visualization plots for given samples."""
    print(f"\nSaving {len(samples_for_viz)} sample visualizations (including uncertainty) to the 'visualizations' directory...")
    for sample_data in samples_for_viz:
        visualize_segmentation(
            sample_data['original_img_path'],
            sample_data['gt_mask_np'],
            sample_data['pred_mask_np'],
            sample_data['uncertainty_map_np'],
            sample_data['dice'],
            sample_data['iou']
        )
    print("Visualization saving complete. Check the 'visualizations' folder.")


# --- Main Evaluation Logic ---
def main():
    """Main function to load predictions, ground truth, and calculate evaluation metrics."""
    setup_visualizations_directory()

    # --- THIS IS THE ADDED DEBUG LINE ---
    print(f"DEBUG: Attempting to load predictions from: {predictions_path.resolve()}")
    # --- END ADDED DEBUG LINE ---

    if not predictions_path.exists():
        print(f"Error: Predictions file not found at {predictions_path}. "
              "Please run train_model.py first to generate it.")
        sys.exit(1)

    stacked_predictions_tensor = torch.load(predictions_path)
    print(f"Loaded predicted masks tensor with shape: {stacked_predictions_tensor.shape}")

    prediction_decoder1_tensor = stacked_predictions_tensor[:, 0, :, :, :]
    prediction_decoder2_tensor = stacked_predictions_tensor[:, 1, :, :, :]

    transform_eval = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    try:
        evaluation_dataset = TestEvaluationDataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            transform=transform_eval
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure 'test_image_dir' and 'test_mask_dir' paths are correct "
              "and contain the relevant files and naming conventions.")
        sys.exit(1)

    num_viz_samples = 5
    all_dice_scores, all_iou_scores, samples_for_viz = process_test_samples(
        evaluation_dataset, prediction_decoder1_tensor, prediction_decoder2_tensor, num_viz_samples
    )

    print_summary_results(all_dice_scores, all_iou_scores)
    generate_visualizations(samples_for_viz)


if __name__ == '__main__':
    main()
