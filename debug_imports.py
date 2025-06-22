import sys
import os
print("DEBUG: os and sys imported.", flush=True)

from pathlib import Path
print("DEBUG: pathlib imported.", flush=True)

import matplotlib.pyplot as plt
print("DEBUG: matplotlib.pyplot imported.", flush=True)

import torch
print("DEBUG: torch imported.", flush=True)

from torchvision import transforms
print("DEBUG: torchvision.transforms imported.", flush=True)

from PIL import Image
print("DEBUG: PIL.Image imported.", flush=True)

print("DEBUG: All core imports successful.", flush=True)

if __name__ == "__main__":
    print("DEBUG: Script finished successfully.", flush=True)
