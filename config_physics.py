"""
Optimized Configuration for Galaxy CGAN - Retrain from Scratch
Adjusted settings for better feature learning and more realistic results
"""
import torch

# Data paths
DATA_PATH = "data/gz2_hart16.csv"
IMAGE_PATH = "data/images_gz2/images"

# Model parameters - OPTIMIZED FOR BETTER QUALITY
IMAGE_SIZE = 64
BATCH_SIZE = 64  # Reduced from 96 for more stable gradients
EPOCHS = 150  # Increased from 100 for better convergence
LR = 0.00015  # Slightly lower for more stable training
BETA1 = 0.5
BETA2 = 0.999

# Architecture
NOISE_DIM = 100
NUM_CLASSES = 4  # Spiral, Elliptical, Merger, Edge-on
NUM_PHYSICAL_ATTRS = 4  # size, brightness, ellipticity, redshift
CONDITION_DIM = NUM_CLASSES + NUM_PHYSICAL_ATTRS  # 8

# Physics-aware loss - OPTIMIZED
USE_PHYSICS_LOSS = True
LAMBDA_PHYSICS = 3.0  # Balanced: not too strong (5.0) or weak (2.0)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
SAMPLE_DIR = "samples"
CHECKPOINT_DIR = "checkpoints"
PHYSICS_PREDICTOR_PATH = "checkpoints/physics_predictor.pth"

# Training - OPTIMIZED
SAVE_INTERVAL = 10
PRETRAIN_PHYSICS_EPOCHS = 30  # Increased from 20 for better physics predictor

# Advanced training settings
GRADIENT_CLIP = 1.0  # Prevent exploding gradients
LABEL_SMOOTHING_REAL = 0.9  # Real labels
LABEL_SMOOTHING_FAKE = 0.1  # Fake labels

"""
KEY CHANGES FROM ORIGINAL:
1. BATCH_SIZE: 96 → 64 (more stable gradients, better learning)
2. EPOCHS: 100 → 150 (more time to learn features)
3. LR: 0.0002 → 0.00015 (slower, more stable)
4. LAMBDA_PHYSICS: 5.0 → 3.0 (balanced - realistic images + attribute control)
5. PRETRAIN_PHYSICS_EPOCHS: 20 → 30 (better physics predictor)

EXPECTED IMPROVEMENTS:
- More stable training (no mode collapse)
- Better feature learning (slower LR, more epochs)
- More realistic images (balanced physics loss)
- Better convergence (smaller batch size)

TRAINING TIME (RTX 3070):
- Physics pre-training: ~25 minutes (30 epochs)
- GAN training: ~4 hours (150 epochs)
- Total: ~4.5 hours
"""
