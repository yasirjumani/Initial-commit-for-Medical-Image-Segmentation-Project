"""This script loads predictions and visualizes them."""

import os
import sys
from pathlib import Path

# Add the try-except block here for PIL import
try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is not installed. Please install it using 'pip install Pillow'")
    sys.exit(1) # Use sys.exit(1) for consistent exit behavior

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

# pylint: disable=C0301,C0303,C0304,R0914,E0401,E1101,W0718,W0611

# --- Configuration (Make sure these paths match your setup) ---
# Define the output directory where predictions are saved by train_model.py
output_dir = Path('output')
predictions_path = output_dir / 'test_predictions.pth'

# Define the path to your test images (this should match what you used in train_model.py)
# Path string split for line length
test_image_path = Path('/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/'
                       'ISIC2018_Task1-2_Test_Input')

# --- Data Loading and Setup ---

# Load the predictions tensor
if not predictions_path.exists():
    # Error message split for line length
    print(f"Error: Predictions file not found at {predictions_path}. "
          "Please run train_model.py first to generate it.")
    sys.exit(1) # Using sys.exit() as recommended

test_predictions = torch.load(predictions_path)
print(f"Loaded predictions tensor with shape: {test_predictions.shape}")

# Image transformations for loading original test images for display
# Only ToTensor is used here to get a float tensor, no normalization for display
transform_display_img = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

class SimpleImageDataset(torch.utils.data.Dataset):
    """A simple dataset class to load images for display."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # Ensure the image directory exists - condition split for line length
        if not self.image_dir.exists() or \
           not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.transform = transform
        # List comprehension split for line length
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        if not self.image_paths:
            # Warning message split for line length
            print(f"Warning: No images found in {self.image_dir}. "
                  "Cannot display original images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path # Return image tensor and its original path for identification

test_image_ds = SimpleImageDataset(
    image_dir=test_image_path,
    transform=transform_display_img
)

# --- Visualization Code ---
# Show up to 5 samples, or fewer if less are available
num_samples_to_show = min(5, len(test_predictions))

if num_samples_to_show > 0:
    print("\nDisplaying sample predictions...")
    plt.figure(figsize=(15, num_samples_to_show * 5))

    # Using enumerate as recommended by Pylint
    for i, (original_image_tensor, original_img_path) in enumerate(test_image_ds):
        if i >= num_samples_to_show: # Only show up to num_samples_to_show
            break

        # Remove channel dimension (1, 256, 256) -> (256, 256)
        predicted_mask_tensor = test_predictions[i].squeeze(0)

        # Convert original image tensor to format for matplotlib (C, H, W) -> (H, W, C)
        original_image_display = original_image_tensor.permute(
            1, 2, 0).numpy()

        # Threshold the predicted mask to make it binary (0 or 1)
        binary_predicted_mask = (predicted_mask_tensor > 0.5).float().numpy()

        # Plot Original Image
        plt.subplot(num_samples_to_show, 2, i * 2 + 1)
        plt.imshow(original_image_display)
        # Title string split for line length
        plt.title(f"Original Image: {Path(original_img_path).name}\n"
                  f"(Sample {i+1})")
        plt.axis('off')

        # Plot Predicted Mask
        plt.subplot(num_samples_to_show, 2, i * 2 + 2)
        plt.imshow(binary_predicted_mask, cmap='gray') # Use 'gray' colormap for binary masks
        plt.title("Predicted Mask (Threshold > 0.5)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No test samples available to display.")


# --- (Optional) Save predicted masks as image files ---
save_masks_dir = output_dir / "predicted_masks"
save_masks_dir.mkdir(exist_ok=True)

# Get original filenames from the SimpleImageDataset
test_image_filenames = [Path(p).name for p in test_image_ds.image_paths]

if test_predictions.shape[0] > 0:
    # Print message split for line length
    print(f"\nSaving all predicted masks as images to {save_masks_dir}...")
    # Using enumerate as recommended by Pylint
    for i, predicted_mask_tensor in enumerate(test_predictions):
        predicted_mask_tensor = predicted_mask_tensor.squeeze(0) # Remove channel dimension
        binary_predicted_mask_np = (predicted_mask_tensor > 0.5).cpu().numpy() # Get binary 0 or 1
        # Scale to 0-255 and convert to PIL Image - moved comment to new line
        # Line too long: Split chained calls
        predicted_mask_pil = Image.fromarray(
            (binary_predicted_mask_np * 255).astype('uint8'))

        # Save with original filename stem + "_predicted_mask.png"
        original_filename_stem = Path(test_image_filenames[i]).stem
        predicted_mask_pil.save(save_masks_dir / f"{original_filename_stem}_predicted_mask.png")
    print(f"All predicted masks saved as images to {save_masks_dir}")
else:
    print("No predictions to save as images.")


# --- Further Analysis Notes ---
print("\nFurther analysis notes:")
print("- The values in 'test_predictions.pth' are probabilities (0-1).")
print("- You can adjust the threshold (currently 0.5) for binary masks based on desired "
      "precision/recall.")
print("- To calculate quantitative metrics (e.g., Dice score, IoU), you would need the "
      "ground truth masks for your test set.")
print("  Your current test data setup is for unannotated data (`mask_dir=None`).")
print("  If you have a separate annotated test set, you'd load its ground truth masks and compare.")
