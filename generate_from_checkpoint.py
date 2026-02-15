"""
Generate samples from a saved checkpoint
"""

import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

from config import *
from gan_model import Generator


# Generator imported from gan_model


def generate_samples(checkpoint_path, num_samples=16, output_name="generated_samples.png"):
    """Generate samples from a checkpoint"""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Initialize generator
    generator = Generator(NOISE_DIM, CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    print(f"Checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Generating {num_samples} samples...")
    
    with torch.no_grad():
        # Create random noise
        noise = torch.randn(num_samples, NOISE_DIM, device=DEVICE)
        
        # Create varied conditions (cycle through all 4 morphology classes)
        conditions = []
        class_names = ["Spiral", "Elliptical", "Merger", "Edge-on"]
        
        for i in range(num_samples):
            # Morphology (one-hot)
            morph = torch.zeros(4)
            morph[i % 4] = 1.0
            
            # Physical attributes (random)
            size = torch.rand(1) * 0.7 + 0.3  # [0.3, 1.0]
            brightness = torch.rand(1) * 0.8 + 0.2  # [0.2, 1.0]
            ellipticity = torch.rand(1) * 0.9  # [0.0, 0.9]
            redshift = torch.rand(1) * 0.5  # [0.0, 0.5]
            
            condition = torch.cat([morph, size, brightness, ellipticity, redshift])
            conditions.append(condition)
            
            # Print condition for this sample
            morph_class = class_names[i % 4]
            print(f"  Sample {i+1}: {morph_class}, S={size.item():.2f}, B={brightness.item():.2f}, "
                  f"E={ellipticity.item():.2f}, Z={redshift.item():.2f}")
        
        conditions = torch.stack(conditions).to(DEVICE)
        
        # Generate images
        fake_images = generator(noise, conditions)
        
        # Save grid
        os.makedirs("generated", exist_ok=True)
        output_path = f"generated/{output_name}"
        vutils.save_image(fake_images, output_path, normalize=True, nrow=4)
        
        print(f"\n✓ Saved {num_samples} samples to {output_path}")
        print(f"  Grid layout: 4x4 (rows cycle through: Spiral, Elliptical, Merger, Edge-on)")


if __name__ == "__main__":
    import sys
    
    # Default to checkpoint 10
    checkpoint_epoch = 10
    
    # Allow command line argument
    if len(sys.argv) > 1:
        checkpoint_epoch = int(sys.argv[1])
    
    checkpoint_path = f"checkpoints/checkpoint_epoch_{checkpoint_epoch}.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        if os.path.exists("checkpoints"):
            checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("checkpoint_epoch_")]
            for cp in sorted(checkpoints):
                print(f"  {cp}")
        else:
            print("  No checkpoints directory found")
    else:
        output_name = f"samples_epoch_{checkpoint_epoch}.png"
        generate_samples(checkpoint_path, num_samples=16, output_name=output_name)
