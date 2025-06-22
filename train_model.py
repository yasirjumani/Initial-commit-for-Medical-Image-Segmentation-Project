"""
Semi-Supervised Medical Image Segmentation Training Script

This script implements a semi-supervised learning approach for medical image segmentation using:
- Explicit consistency regularization for unlabeled data
- Multiple decoders for improved segmentation and ambiguity awareness
- Combined Dice and BCE loss functions
"""

# pylint: disable=C0301,C0303,C0304,R0914,E0401,E1101,W0718,W0611,R0902,R0913,R0917,R0915

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is not installed. Please install it using 'pip install Pillow'")
    raise

import os
import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt # Not strictly used in this version for plots, but kept
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

class MultiDecoderCNN(nn.Module):
    """CNN with multiple decoders for semi-supervised learning.

    Attributes:
        encoder: Shared encoder network
        decoder1: First decoder head
        decoder2: Second decoder head
    """

    def __init__(self, num_classes=1):
        """Initialize the MultiDecoderCNN.

        Args:
            num_classes: Number of output classes/channels
        """
        super().__init__()
        # Example encoder (replace with your actual architecture)
        # Input: 3x256x256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # Output: 64x256x256
            nn.ReLU(),
            nn.MaxPool2d(2),                                     # Output: 64x128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: 128x128x128
            nn.ReLU(),
            nn.MaxPool2d(2)                                      # Output: 128x64x64 (Encoder's feature map size)
        )

        # Multiple decoders
        # Needs to upsample from 128x64x64 back to 1x256x256
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # Upsample 1: 64x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Upsample 2: 32x256x256
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)             # Final convolution: 1x256x256
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # Upsample 1: 64x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Upsample 2: 32x256x256
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)             # Final convolution: 1x256x256
        )

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Tuple of outputs from both decoders
        """
        features = self.encoder(x)
        out1 = self.decoder1(features)
        out2 = self.decoder2(features)
        return out1, out2

def combined_loss(pred, target):
    """Calculate combined Dice and Binary Cross-Entropy loss for segmentation.

    Args:
        pred: Model predictions (logits)
        target: Ground truth masks

    Returns:
        Combined loss value (Dice + BCE)
    """
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)

    # Dice loss calculation
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum()
    union = pred_sigmoid.sum() + target.sum()
    dice_loss = 1 - (2. * intersection + 1) / (union + 1)

    return bce_loss + dice_loss

# Dataset paths
image_dataset_path = Path(
    '/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/ISIC2018_Task1-2_Training_Input'
)
mask_dataset_path = Path(
    '/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/ISIC2018_Task1_Training_GroundTruth'
)
validation_image_path = Path(
    '/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/ISIC2018_Task1-2_Validation_Input'
)
validation_mask_path = Path(
    '/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/ISIC2018_Task1_Validation_GroundTruth'
)
test_image_path = Path(
    '/Users/yasirahmed/Desktop/medical_image_segmentation/dataset/ISIC2018_Task1-2_Test_Input'
)
# For semi-supervised learning, we'll treat the test_image_path as the source
# for unlabeled data. If you have a separate unlabeled dataset, point this to it.
unlabeled_image_path = test_image_path


# Create output directory
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

class CustomImageSegmentationDataset(torch.utils.data.Dataset):
    """Custom dataset loader for medical image segmentation with pseudo-labeling support."""

    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None,
                 augment=False):
        """Initialize dataset with image and mask directories.

        Args:
            image_dir: Path to image directory
            mask_dir: Path to mask directory (optional)
            transform: Transformations for images
            mask_transform: Transformations for masks
            augment: Whether to apply additional data augmentations (flips, rotations)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment

        print(f"DEBUG: Initializing dataset with image_dir={image_dir} and mask_dir={mask_dir}")

        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if mask_dir and (not mask_dir.exists() or not mask_dir.is_dir()):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png')) and not f.startswith('.')
        ])

        if mask_dir:
            self.mask_paths = sorted([
                os.path.join(mask_dir, f)
                for f in os.listdir(mask_dir)
                if f.lower().endswith('.png') and '_segmentation' in f.lower() and not f.startswith('.')
            ])
            print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
        else:
            self.mask_paths = None

        self.aligned_paths = []
        if mask_dir:
            for img_path in self.image_paths:
                img_filename = Path(img_path).stem
                expected_mask_filename = img_filename + '_segmentation.png'
                mask_path = os.path.join(mask_dir, expected_mask_filename)
                if os.path.exists(mask_path):
                    self.aligned_paths.append((img_path, mask_path))

            if not self.aligned_paths:
                raise RuntimeError(f"No aligned image-mask pairs found. Checked {len(self.image_paths)} images in {image_dir}.")
            print(f"Successfully aligned {len(self.aligned_paths)} image-mask pairs")
        else:
            # For test or unlabeled data without masks
            self.aligned_paths = [(img_path, None) for img_path in self.image_paths]

    def __len__(self):
        """Return the number of image-mask pairs."""
        return len(self.aligned_paths)

    def __getitem__(self, idx):
        """Get an image-mask pair by index.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, mask) or (image, dummy_tensor_for_mask)
        """
        img_path, mask_path = self.aligned_paths[idx]
        img = Image.open(img_path).convert('RGB')

        mask = None
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')

        # Apply geometric augmentations if self.augment is True AND a mask exists
        # This ensures augmentations are applied consistently to image and mask
        if self.augment and mask is not None:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = transforms.RandomRotation.get_params([-15, 15])
            img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # Apply standard image transforms (including color jitter if transform_train_img)
        if self.transform:
            img = self.transform(img)

        # For unlabeled data, return a dummy tensor for the mask instead of None.
        if mask_path is None:
            return img, torch.zeros((1, 256, 256), dtype=torch.float32)

        # Apply standard mask transforms
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return img, mask

# Image transformations for training
transform_train_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mask transformations for training
transform_train_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Transformations for validation/test (no augmentation applied)
transform_eval_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_eval_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_datasets():
    """Load and validate all datasets.

    Returns:
        Tuple of (train_ds, validation_ds, unlabeled_ds, test_ds) datasets
    """
    print("\nLoading datasets...")
    try:
        # Labeled Training Dataset
        train_ds = CustomImageSegmentationDataset(
            image_dir=image_dataset_path,
            mask_dir=mask_dataset_path,
            transform=transform_train_img,
            mask_transform=transform_train_mask,
            augment=True
        )

        # Validation Dataset
        validation_ds = CustomImageSegmentationDataset(
            image_dir=validation_image_path,
            mask_dir=validation_mask_path,
            transform=transform_eval_img,
            mask_transform=transform_eval_mask,
            augment=False
        )

        # Unlabeled Dataset for Semi-Supervised Learning
        # This dataset is for training, so it uses transform_train_img and augment=True.
        # It's crucial that mask_dir is None here.
        unlabeled_ds = CustomImageSegmentationDataset(
            image_dir=unlabeled_image_path,
            mask_dir=None, # Crucially, no masks for this dataset
            transform=transform_train_img, # Apply same transformations as labeled images
            augment=True # Apply augmentations to unlabeled data as well
        )

        # Test Dataset (for final inference, without masks, no augmentation)
        test_ds = CustomImageSegmentationDataset(
            image_dir=test_image_path,
            transform=transform_eval_img,
            augment=False
        )

        print("Datasets loaded successfully")
        return train_ds, validation_ds, unlabeled_ds, test_ds

    except Exception as e:
        print(f"\nDataset loading failed: {e}")
        raise

# Helper function to run validation
def evaluate_model_on_val_set(model, val_loader, device):
    """Evaluates the model on the validation set."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1, outputs2 = model(inputs)
            # For validation, we typically only care about supervised loss against ground truth
            # Sum losses from both decoders, just like in training supervised loss
            val_loss += combined_loss(outputs1, targets).item() + combined_loss(outputs2, targets).item()
    return val_loss / len(val_loader)


def train_model():
    """Train the segmentation model with consistency regularization.

    Returns:
        Trained model
    """
    # --- Pylint R0915 (too-many-statements) note: ---
    # This function is still quite long. For larger projects, you would extract
    # the training loop logic (e.g., `train_one_epoch`), and the validation logic
    # into separate functions to improve readability and maintainability.
    # For a course project, this might be acceptable given the complexity.

    print("\nStarting training...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load datasets
    train_ds, validation_ds, unlabeled_ds, _ = load_datasets()

    # Create data loaders
    batch_size = 16
    num_workers = 0 # Explicitly set num_workers=0 to avoid multiprocessing issues on MPS/Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=num_workers)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(validation_ds)}, Unlabeled samples: {len(unlabeled_ds)}")

    model = MultiDecoderCNN(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    consistency_loss_fn = nn.MSELoss() # For consistency regularization between decoder outputs

    num_epochs = 50 # Increased epochs for better training convergence with semi-supervised
    lambda_unlabeled_max = 0.5 # Maximum weight for unlabeled consistency loss
    ramp_up_epochs = 10 # Number of epochs to ramp up the unlabeled loss weight

    for epoch in range(num_epochs):
        model.train()
        running_labeled_loss = 0.0
        running_unlabeled_consistency_loss = 0.0 # Total unlabeled consistency loss for reporting

        # Create iterators for loaders to handle different dataset sizes
        labeled_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader)

        # Iterate as many times as the largest loader, resetting the smaller one
        num_batches_per_epoch = max(len(train_loader), len(unlabeled_loader))

        # Calculate current lambda_unlabeled (ramp-up schedule)
        # This weight gradually increases over the first 'ramp_up_epochs'
        current_consistency_weight = lambda_unlabeled_max
        if epoch < ramp_up_epochs:
            current_consistency_weight = lambda_unlabeled_max * (epoch / ramp_up_epochs)

        for _ in range(num_batches_per_epoch): # Use _ for unused loop variable, addressing W0612
            optimizer.zero_grad()

            # --- Labeled Data Step ---
            inputs_labeled, targets_labeled = None, None
            try:
                inputs_labeled, targets_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_loader) # Reset labeled data iterator
                inputs_labeled, targets_labeled = next(labeled_iter)

            inputs_labeled, targets_labeled = inputs_labeled.to(device), targets_labeled.to(device)

            outputs1_labeled, outputs2_labeled = model(inputs_labeled)
            loss_labeled_decoder1 = combined_loss(outputs1_labeled, targets_labeled)
            loss_labeled_decoder2 = combined_loss(outputs2_labeled, targets_labeled)
            supervised_loss = loss_labeled_decoder1 + loss_labeled_decoder2
            running_labeled_loss += supervised_loss.item()

            # --- Unlabeled Data Step (Consistency Regularization) ---
            inputs_unlabeled = None
            try:
                inputs_unlabeled, _ = next(unlabeled_iter) # Mask is dummy, ignore it
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader) # Reset unlabeled data iterator
                inputs_unlabeled, _ = next(unlabeled_iter)

            inputs_unlabeled = inputs_unlabeled.to(device)

            outputs1_unlabeled, outputs2_unlabeled = model(inputs_unlabeled)

            # Consistency Loss: Minimize the difference between decoder outputs for unlabeled data
            # Applying sigmoid to get probabilities for consistency calculation
            consistency_loss = consistency_loss_fn(
                torch.sigmoid(outputs1_unlabeled),
                torch.sigmoid(outputs2_unlabeled)
            )

            unlabeled_total_loss = current_consistency_weight * consistency_loss
            running_unlabeled_consistency_loss += unlabeled_total_loss.item()

            # --- Total Loss for Backpropagation ---
            total_loss = supervised_loss + unlabeled_total_loss
            total_loss.backward()
            optimizer.step()

        # Validation phase
        avg_val_loss = evaluate_model_on_val_set(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Labeled Sup. Loss: {running_labeled_loss/len(train_loader):.4f} - "
              f"Train Unlabeled Cons. Loss: {running_unlabeled_consistency_loss/len(unlabeled_loader):.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"Consistency Weight: {current_consistency_weight:.4f}")

    torch.save(model.state_dict(), output_dir / 'trained_model.pth')
    print(f"Model saved to {output_dir / 'trained_model.pth'}")
    return model

def inference_on_test_data(model, test_loader, device):
    """Perform inference on the test dataset.

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
    """
    model.eval()
    all_predictions_decoder1 = []
    all_predictions_decoder2 = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0] # Get the images from the batch

            inputs = inputs.to(device)
            outputs1, outputs2 = model(inputs)

            # Apply sigmoid to get probabilities
            predictions1 = torch.sigmoid(outputs1)
            predictions2 = torch.sigmoid(outputs2)

            all_predictions_decoder1.append(predictions1.cpu())
            all_predictions_decoder2.append(predictions2.cpu())

    all_predictions_decoder1 = torch.cat(all_predictions_decoder1)
    all_predictions_decoder2 = torch.cat(all_predictions_decoder2)

    # Stack predictions from both decoders along a new dimension
    # Resulting shape will be [N, 2, 1, H, W]
    # N = number of samples, 2 = number of decoders, 1 = channels (grayscale mask), H, W = height, width
    stacked_predictions = torch.stack([all_predictions_decoder1, all_predictions_decoder2], dim=1)

    # Save predictions as a tensor file
    torch.save(stacked_predictions, output_dir / 'test_predictions.pth')
    print(f"Test predictions (from both decoders) saved to {output_dir / 'test_predictions.pth'}")

def main():
    """Main entry point for the training script."""
    print("Starting script execution...")
    try:
        # Determine device for model loading/training/inference
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model_path = output_dir / 'trained_model.pth'

        # Initialize model first, then load state_dict or train
        trained_model = MultiDecoderCNN(num_classes=1).to(device)

        if model_path.exists():
            print(f"Loading pre-trained model from {model_path}...")
            # Load state_dict, ensuring it's mapped to the correct device
            trained_model.load_state_dict(torch.load(model_path, map_location=device))
            trained_model.eval() # Set to evaluation mode for inference
            print("Model loaded successfully. Skipping training phase.")
        else:
            print("No pre-trained model found. Starting training...")
            trained_model = train_model() # This function will save the model upon completion

        # Load all datasets again to get the test_ds for inference
        # The return tuple is (train_ds, validation_ds, unlabeled_ds, test_ds)
        _, _, _, test_ds = load_datasets()
        # Ensure test_loader also uses num_workers=0 and batch_size=16
        test_loader = DataLoader(test_ds, batch_size=16, num_workers=0)

        # Perform inference
        inference_on_test_data(trained_model, test_loader, device)

    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {e}")
        sys.exit(1)
    print("Script completed successfully")

if __name__ == "__main__":
    main()
