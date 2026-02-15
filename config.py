"""
Configuration for Physics-Aware Galaxy CGAN
"""
import torch

# Data paths
DATA_PATH = "data/gz2_hart16.csv"
IMAGE_PATH = "data/images_gz2/images"

# Model parameters
IMAGE_SIZE = 64
BATCH_SIZE = 96
EPOCHS = 100
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Architecture
NOISE_DIM = 100
NUM_CLASSES = 4  # Spiral, Elliptical, Merger, Edge-on
NUM_PHYSICAL_ATTRS = 4  # size, brightness, ellipticity, redshift
CONDITION_DIM = NUM_CLASSES + NUM_PHYSICAL_ATTRS  # 8

# Physics-aware loss
USE_PHYSICS_LOSS = True
LAMBDA_PHYSICS = 5.0  # Weight for physics loss (lower than original to avoid overwhelming GAN loss)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
SAMPLE_DIR = "samples"
CHECKPOINT_DIR = "checkpoints"
PHYSICS_PREDICTOR_PATH = "checkpoints/physics_predictor.pth"

# Training
SAVE_INTERVAL = 10
PRETRAIN_PHYSICS_EPOCHS = 20  # Epochs to pre-train physics predictor
