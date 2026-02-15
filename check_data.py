"""
Quick diagnostic script to check if real galaxy images are loading correctly
"""

import torch
from cgan_galaxy import GalaxyDataset
from config import *
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
print("Loading dataset...")
dataset = GalaxyDataset(DATA_PATH, IMAGE_PATH, IMAGE_SIZE)

print(f"\nDataset size: {len(dataset)}")

# Get a few samples
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Real Galaxy Images from Dataset - 4 Classes', fontsize=16)

for i in range(8):
    img, condition = dataset[i * 1000]  # Sample every 1000th image
    
    # Convert from [-1, 1] to [0, 1] for display
    img_display = (img.permute(1, 2, 0).numpy() + 1) / 2
    img_display = np.clip(img_display, 0, 1)
    
    ax = axes[i // 4, i % 4]
    ax.imshow(img_display)
    ax.axis('off')
    
    # Show condition (4 classes + 4 attributes)
    class_names = ["Spiral", "Elliptical", "Merger", "Edge-on"]
    morph_idx = torch.argmax(condition[:4]).item()
    morph_name = class_names[morph_idx]
    size, brightness, ellipticity, redshift = condition[4:8]
    ax.set_title(f"{morph_name}\nS:{size:.2f} B:{brightness:.2f}\nE:{ellipticity:.2f} Z:{redshift:.2f}", 
                 fontsize=7)

plt.tight_layout()
plt.savefig('real_images_check.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved real_images_check.png")
print("\nNow showing all 4 classes:")
print("  Spiral, Elliptical, Merger, Edge-on")
print("  S=size, B=brightness, E=ellipticity, Z=redshift")
