"""
Quick test script to verify CGAN implementation
Tests model initialization and forward pass without training
"""

import torch
from gan_model import Generator, Discriminator
from config import *

def test_models():
    """Test model initialization and forward pass"""
    print("=" * 60)
    print("Testing Minimal CGAN Implementation")
    print("=" * 60)
    
    # Check device
    print(f"\n1. Device Check:")
    print(f"   Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize models
    print(f"\n2. Model Initialization:")
    generator = Generator(NOISE_DIM, CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    discriminator = Discriminator(CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"   Generator parameters: {g_params:,}")
    print(f"   Discriminator parameters: {d_params:,}")
    print(f"   Total parameters: {g_params + d_params:,}")
    
    # Test forward pass
    print(f"\n3. Forward Pass Test:")
    batch_size = 4
    
    # Create dummy inputs
    noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
    condition = torch.randn(batch_size, CONDITION_DIM, device=DEVICE)
    
    # Generator forward
    print(f"   Input noise shape: {noise.shape}")
    print(f"   Input condition shape: {condition.shape}")
    
    fake_imgs = generator(noise, condition)
    print(f"   Generated image shape: {fake_imgs.shape}")
    print(f"   Image value range: [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]")
    
    # Discriminator forward
    disc_output = discriminator(fake_imgs, condition)
    print(f"   Discriminator output shape: {disc_output.shape}")
    print(f"   Output value range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")
    
    # Memory usage
    if DEVICE == "cuda":
        print(f"\n4. Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ready to train.")
    print("=" * 60)
    print("\nTo start training, run:")
    print("  python galaxy.py")
    print()

if __name__ == "__main__":
    test_models()
