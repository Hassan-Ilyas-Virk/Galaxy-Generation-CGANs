"""
Physics-Aware Conditional GAN for Galaxy Generation
Complete implementation with Physics Predictor and Physics Loss
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

from config_physics import *
import argparse
from diffusion_model import GaussianDiffusion, ContextUnet
from gan_model import Generator, Discriminator
from physics_model import PhysicsPredictor


# ============================================================================
# Dataset
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
        
        # 4-Class Morphology with better thresholds
        merger_mask = df['t08_odd_feature_a24_merger_weighted_fraction'] > 0.4
        edgeon_mask = (~merger_mask) & (df['t02_edgeon_a04_yes_weighted_fraction'] > 0.5)
        smooth = df['t01_smooth_or_features_a01_smooth_weighted_fraction']
        features = df['t01_smooth_or_features_a02_features_or_disk_weighted_fraction']
        # Stronger thresholds for better classification
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
        
        # Condition vector: [4 morph classes, 4 physical attrs]
        morph_onehot = torch.zeros(4)
        morph_onehot[self.morphology[idx]] = 1.0
        physical_attrs = torch.tensor([self.size[idx], self.brightness[idx], 
                                       self.ellipticity[idx], self.redshift[idx]], dtype=torch.float32)
        condition = torch.cat([morph_onehot, physical_attrs])
        
        return img, condition


# ============================================================================
# Models
# ============================================================================

# Models are now imported from gan_model.py and physics_model.py


# ============================================================================
# Training Functions
# ============================================================================

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_generator_output(generator):
    for module in reversed(list(generator.modules())):
        if isinstance(module, nn.ConvTranspose2d) and module.out_channels == 3:
            nn.init.constant_(module.bias.data, -1.0)
            print("✓ Generator initialized for black background")
            break


def pretrain_physics_predictor(dataset, device, epochs=20):
    """Pre-train physics predictor on real galaxy images"""
    print("\n" + "="*60)
    print("PRE-TRAINING PHYSICS PREDICTOR")
    print("="*60)
    
    predictor = PhysicsPredictor(IMAGE_SIZE, NUM_PHYSICAL_ATTRS).to(device)
    predictor.apply(weights_init)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for imgs, conditions in pbar:
            imgs = imgs.to(device)
            # Extract physical attributes (last 4 values of condition)
            true_attrs = conditions[:, 4:].to(device)
            
            optimizer.zero_grad()
            pred_attrs = predictor(imgs)
            loss = criterion(pred_attrs, true_attrs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    # Save predictor
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(predictor.state_dict(), PHYSICS_PREDICTOR_PATH)
    print(f"✓ Physics predictor saved to {PHYSICS_PREDICTOR_PATH}\n")
    
    return predictor


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


def train_with_physics_loss():
    """Main training loop with physics-aware loss"""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = GalaxyDataset(DATA_PATH, IMAGE_PATH, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # Pre-train or load physics predictor
    if os.path.exists(PHYSICS_PREDICTOR_PATH):
        print(f"Loading pre-trained physics predictor from {PHYSICS_PREDICTOR_PATH}")
        physics_predictor = PhysicsPredictor(IMAGE_SIZE, NUM_PHYSICAL_ATTRS).to(DEVICE)
        physics_predictor.load_state_dict(torch.load(PHYSICS_PREDICTOR_PATH))
    else:
        physics_predictor = pretrain_physics_predictor(dataset, DEVICE, PRETRAIN_PHYSICS_EPOCHS)
    
    physics_predictor.eval()  # Freeze physics predictor during GAN training
    for param in physics_predictor.parameters():
        param.requires_grad = False
    
    # Initialize GAN
    print("\nInitializing GAN models...")
    generator = Generator(NOISE_DIM, CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    discriminator = Discriminator(CONDITION_DIM, IMAGE_SIZE).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    init_generator_output(generator)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    criterion_gan = nn.BCELoss()
    criterion_physics = nn.MSELoss()
    
    print(f"\nStarting GAN training with physics loss (λ={LAMBDA_PHYSICS})...")
    print(f"Device: {DEVICE}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}\n")
    
    for epoch in range(EPOCHS):
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
            
            # GAN loss
            g_loss_gan = criterion_gan(fake_output, torch.ones(batch_size, 1, device=DEVICE))
            
            # Physics loss: predicted attributes should match requested attributes
            if USE_PHYSICS_LOSS:
                pred_attrs = physics_predictor(fake_imgs)
                true_attrs = conditions[:, 4:]  # Extract physical attributes from condition
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
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    print("\n✓ Training complete!")


def train_diffusion_with_physics_loss():
    """Main training loop for Diffusion Model with physics-aware loss"""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = GalaxyDataset(DATA_PATH, IMAGE_PATH, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # Pre-train or load physics predictor
    if os.path.exists(PHYSICS_PREDICTOR_PATH):
        print(f"Loading pre-trained physics predictor from {PHYSICS_PREDICTOR_PATH}")
        physics_predictor = PhysicsPredictor(IMAGE_SIZE, NUM_PHYSICAL_ATTRS).to(DEVICE)
        physics_predictor.load_state_dict(torch.load(PHYSICS_PREDICTOR_PATH))
    else:
        physics_predictor = pretrain_physics_predictor(dataset, DEVICE, PRETRAIN_PHYSICS_EPOCHS)
    
    physics_predictor.eval()
    for param in physics_predictor.parameters():
        param.requires_grad = False
    
    # Initialize Diffusion Model
    print("\nInitializing Diffusion Model...")
    diffusion = GaussianDiffusion(timesteps=100)
    model = ContextUnet(in_channels=3, n_feat=128, n_classes=NUM_CLASSES, n_phys=NUM_PHYSICAL_ATTRS).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_physics = nn.MSELoss()
    
    print(f"\nStarting Diffusion training with physics loss (λ={LAMBDA_PHYSICS})...")
    print(f"Device: {DEVICE}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}\n")
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        epoch_p_loss = 0
        
        for real_imgs, conditions in pbar:
            batch_size = real_imgs.size(0)
            real_imgs, conditions = real_imgs.to(DEVICE), conditions.to(DEVICE)
            
            # Extract conditions
            class_labels = conditions[:, :NUM_CLASSES]
            physical_params = conditions[:, NUM_CLASSES:]
            
            optimizer.zero_grad()
            
            # Sample t
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=DEVICE).long()
            
            # Diffusion loss
            diff_loss = diffusion.loss(model, real_imgs, t, class_labels, physical_params)
            
            # Physics loss (on predicted x_0)
            # We need to predict x_0 from the model's noise prediction to apply physics loss
            # This is a bit expensive, so maybe we only do it for small t or periodically?
            # For now, let's do it every step but maybe we can optimize.
            # Actually, let's follow the GAN approach: generate from noise and check physics?
            # But diffusion generates iteratively. 
            # Alternative: Use the predicted x_0 from the current step's noise prediction.
            # x_0 = (x_t - sqrt(1-alpha_cumprod) * noise_pred) / sqrt(alpha_cumprod)
            
            if USE_PHYSICS_LOSS:
                # Get noise prediction again (or reuse if we modify loss function to return it)
                # For simplicity, let's just re-run or rely on the fact that we want the model 
                # to generate physically consistent images.
                # A common way to guide diffusion is classifier guidance or classifier-free guidance.
                # Here we want to enforce physics on the output.
                # Let's try to enforce it on the predicted x_start (denoised version).
                
                noise = torch.randn_like(real_imgs)
                x_noisy = diffusion.q_sample(real_imgs, t, noise)
                predicted_noise = model(x_noisy, t, class_labels, physical_params)
                
                # Reconstruct x_0
                sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / diffusion.alphas_cumprod).to(DEVICE)
                sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / diffusion.alphas_cumprod - 1).to(DEVICE)
                
                sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
                sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
                
                pred_x0 = sqrt_recip_alphas_cumprod_t * x_noisy - sqrt_recipm1_alphas_cumprod_t * predicted_noise
                pred_x0 = torch.clamp(pred_x0, -1, 1) # Clip to valid range
                
                pred_attrs = physics_predictor(pred_x0)
                physics_loss = criterion_physics(pred_attrs, physical_params)
                
                loss = diff_loss + LAMBDA_PHYSICS * physics_loss
            else:
                loss = diff_loss
                physics_loss = torch.tensor(0.0)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_p_loss += physics_loss.item() if USE_PHYSICS_LOSS else 0
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Diff': f'{diff_loss.item():.4f}',
                'Phys': f'{physics_loss.item():.4f}' if USE_PHYSICS_LOSS else '0'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        avg_p = epoch_p_loss / len(dataloader) if USE_PHYSICS_LOSS else 0
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Physics={avg_p:.4f}")
        
        if (epoch + 1) % SAVE_INTERVAL == 0:
            # Save samples
            print("Generating samples...")
            sample_shape = (16, 3, IMAGE_SIZE, IMAGE_SIZE)
            
            # Create random conditions for sampling
            sample_conditions = []
            for i in range(16):
                morph = torch.zeros(4)
                morph[i % 4] = 1.0
                cond = torch.cat([
                    morph,
                    torch.rand(1) * 0.7 + 0.3,
                    torch.rand(1) * 0.8 + 0.2,
                    torch.rand(1) * 0.9,
                    torch.rand(1) * 0.5
                ])
                sample_conditions.append(cond)
            sample_conditions = torch.stack(sample_conditions).to(DEVICE)
            
            sample_class_labels = sample_conditions[:, :NUM_CLASSES]
            sample_physical_params = sample_conditions[:, NUM_CLASSES:]
            
            samples = diffusion.sample(model, sample_shape, sample_class_labels, sample_physical_params, DEVICE)
            vutils.save_image(samples, f"{SAMPLE_DIR}/diffusion_epoch_{epoch+1:03d}.png", normalize=True, nrow=4)
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"{CHECKPOINT_DIR}/diffusion_checkpoint_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    print("\n✓ Diffusion Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Galaxy Generator")
    parser.add_argument("--diffusion", action="store_true", help="Train diffusion model instead of GAN")
    args = parser.parse_args()
    
    if args.diffusion:
        train_diffusion_with_physics_loss()
    else:
        train_with_physics_loss()
