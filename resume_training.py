"""
Resume training from the latest checkpoint
Automatically finds and loads the most recent checkpoint and continues training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import glob

from config_physics import *


# ============================================================================
# Import Models and Dataset
# ============================================================================
from gan_model import Generator, Discriminator
from physics_model import PhysicsPredictor
# ============================================================================

class GalaxyDataset(Dataset):
    """Galaxy Zoo 2 dataset with 4-class morphology and 4 physical attributes"""
    
    def __init__(self, csv_path, image_dir, image_size=64):
        self.image_dir = image_dir
        self.image_size = image_size
        
        df = pd.read_csv(csv_path)
        
        # Load filename mapping
        mapping_path = csv_path.replace('gz2_hart16.csv', 'gz2_filename_mapping.csv')
        if os.path.exists(mapping_path):
            mapping_df = pd.read_csv(mapping_path)
            mapping_df = mapping_df.rename(columns={'objid': 'dr7objid', 'asset_id': 'filename'})
            df = df.merge(mapping_df[['dr7objid', 'filename']], on='dr7objid', how='inner')
            print(f"✓ Loaded {len(df)} galaxies with images")
        
        # 4-Class Morphology
        merger_mask = df['t08_odd_feature_a24_merger_weighted_fraction'] > 0.4
        edgeon_mask = (~merger_mask) & (df['t02_edgeon_a04_yes_weighted_fraction'] > 0.5)
        smooth = df['t01_smooth_or_features_a01_smooth_weighted_fraction']
        features = df['t01_smooth_or_features_a02_features_or_disk_weighted_fraction']
        elliptical_mask = (~merger_mask) & (~edgeon_mask) & (smooth > 0.6) & (smooth > features)
        spiral_mask = (~merger_mask) & (~edgeon_mask) & (features > 0.5)
        
        morphology = np.zeros(len(df), dtype=int)
        morphology[spiral_mask] = 0
        morphology[elliptical_mask] = 1
        morphology[merger_mask] = 2
        morphology[edgeon_mask] = 3
        
        self.filenames = df['filename'].values
        self.morphology = morphology
        
        # Physical attributes
        size = df['total_votes'].values.astype(float)
        self.size = (size - size.min()) / (size.max() - size.min() + 1e-8) * 0.7 + 0.3
        self.brightness = np.clip(smooth.values * 0.8 + 0.2, 0.2, 1.0)
        
        round_frac = df['t07_rounded_a16_completely_round_weighted_fraction'].fillna(0).values
        between_frac = df['t07_rounded_a17_in_between_weighted_fraction'].fillna(0).values
        cigar_frac = df['t07_rounded_a18_cigar_shaped_weighted_fraction'].fillna(0).values
        self.ellipticity = np.clip(round_frac * 0.0 + between_frac * 0.5 + cigar_frac * 0.9, 0.0, 0.9)
        
        np.random.seed(42)
        self.redshift = np.random.uniform(0.0, 0.5, len(df))
        
        print(f"Dataset loaded: Spiral={np.sum(morphology==0)}, Elliptical={np.sum(morphology==1)}, "
              f"Merger={np.sum(morphology==2)}, Edge-on={np.sum(morphology==3)}")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, f"{filename}.jpg")
        
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            img = np.array(img, dtype=np.float32) / 255.0
            if img.std() < 0.01:
                raise ValueError("No variation")
            img = (img - 0.5) / 0.5
            img = torch.FloatTensor(img).permute(2, 0, 1)
        except:
            img = torch.zeros(3, self.image_size, self.image_size)
            center = self.image_size // 2
            img[:, center-2:center+2, center-2:center+2] = 0.5
        
        morph_onehot = torch.zeros(4)
        morph_onehot[self.morphology[idx]] = 1.0
        physical_attrs = torch.tensor([self.size[idx], self.brightness[idx], 
                                       self.ellipticity[idx], self.redshift[idx]], dtype=torch.float32)
        condition = torch.cat([morph_onehot, physical_attrs])
        
        return img, condition


# Models imported from gan_model.py and physics_model.py


def save_samples(generator, epoch, device, num_samples=16):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, NOISE_DIM, device=device)
        conditions = []
        for i in range(num_samples):
            morph = torch.zeros(4)
            morph[i % 4] = 1.0
            condition = torch.cat([
                morph,
                torch.rand(1) * 0.7 + 0.3,
                torch.rand(1) * 0.8 + 0.2,
                torch.rand(1) * 0.9,
                torch.rand(1) * 0.5
            ])
            conditions.append(condition)
        conditions = torch.stack(conditions).to(device)
        fake_images = generator(noise, conditions)
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        vutils.save_image(fake_images, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png", 
                         normalize=True, nrow=4)
    generator.train()


def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest_epoch}.pth")
    
    return latest_checkpoint, latest_epoch


def resume_training():
    """Resume training from the latest checkpoint"""
    
    print("=" * 60)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 60)
    
    # Find latest checkpoint
    checkpoint_info = find_latest_checkpoint()
    if checkpoint_info is None:
        print("❌ No checkpoint found! Please train from scratch using galaxy.py")
        return
    
    checkpoint_path, start_epoch = checkpoint_info
    print(f"✓ Found checkpoint: {checkpoint_path}")
    print(f"✓ Resuming from epoch {start_epoch}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = GalaxyDataset(DATA_PATH, IMAGE_PATH, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # Load physics predictor
    print(f"\nLoading physics predictor from {PHYSICS_PREDICTOR_PATH}")
    physics_predictor = PhysicsPredictor(IMAGE_SIZE, NUM_PHYSICAL_ATTRS).to(DEVICE)
    physics_predictor.load_state_dict(torch.load(PHYSICS_PREDICTOR_PATH))
    physics_predictor.eval()
    for param in physics_predictor.parameters():
        param.requires_grad = False
    
    # Initialize models
    print("\nInitializing models...")
    generator = Generator(NOISE_DIM, CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    discriminator = Discriminator(CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # Load checkpoint
    print(f"\nLoading checkpoint from epoch {start_epoch}...")
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    
    print(f"✓ Models and optimizers loaded successfully")
    
    criterion_gan = nn.BCELoss()
    criterion_physics = nn.MSELoss()
    
    print("\n" + "=" * 60)
    print(f"CONTINUING TRAINING: Epoch {start_epoch + 1} → {EPOCHS}")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Physics Loss Weight: {LAMBDA_PHYSICS}")
    print(f"Remaining Epochs: {EPOCHS - start_epoch}")
    print("=" * 60 + "\n")
    
    # Training loop (continue from start_epoch + 1)
    for epoch in range(start_epoch, EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_d_loss = epoch_g_loss = epoch_p_loss = 0
        
        for real_imgs, conditions in pbar:
            batch_size = real_imgs.size(0)
            real_imgs, conditions = real_imgs.to(DEVICE), conditions.to(DEVICE)
            real_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE) + 0.1
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(real_imgs, conditions)
            d_loss_real = criterion_gan(real_output, real_labels)
            
            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_imgs = generator(noise, conditions)
            fake_output = discriminator(fake_imgs.detach(), conditions)
            d_loss_fake = criterion_gan(fake_output, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator with Physics Loss
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_imgs = generator(noise, conditions)
            fake_output = discriminator(fake_imgs, conditions)
            
            g_loss_gan = criterion_gan(fake_output, torch.ones(batch_size, 1, device=DEVICE))
            
            if USE_PHYSICS_LOSS:
                pred_attrs = physics_predictor(fake_imgs)
                true_attrs = conditions[:, 4:]
                physics_loss = criterion_physics(pred_attrs, true_attrs)
                g_loss = g_loss_gan + LAMBDA_PHYSICS * physics_loss
            else:
                physics_loss = torch.tensor(0.0)
                g_loss = g_loss_gan
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss_gan.item()
            epoch_p_loss += physics_loss.item() if USE_PHYSICS_LOSS else 0
            
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss_gan.item():.3f}',
                'P': f'{physics_loss.item():.3f}' if USE_PHYSICS_LOSS else '0',
                'D_r': f'{real_output.mean().item():.2f}',
                'D_f': f'{fake_output.mean().item():.2f}'
            })
        
        avg_d = epoch_d_loss / len(dataloader)
        avg_g = epoch_g_loss / len(dataloader)
        avg_p = epoch_p_loss / len(dataloader) if USE_PHYSICS_LOSS else 0
        print(f"\nEpoch {epoch+1}: D={avg_d:.4f}, G={avg_g:.4f}, Physics={avg_p:.4f}")
        
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_samples(generator, epoch + 1, DEVICE)
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth")
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    resume_training()
