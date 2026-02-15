# Physics-Aware Galaxy Generation with Conditional GANs

A deep learning framework for generating scientifically valid galaxy images using Conditional Generative Adversarial Networks (CGANs) with physics-aware loss functions. This project enforces astrophysical parameter consistency while generating realistic galaxy morphologies from the Galaxy Zoo 2 dataset.

## Overview

This project addresses a critical gap in astronomical generative AI by ensuring scientific validity while generating realistic galaxy images. Unlike traditional GANs that focus solely on visual realism, our framework integrates a physics predictor network that enforces consistency with astrophysical parameters through MSE-based physics loss.

### Key Features

- **Physics-Aware Loss Framework**: Novel auxiliary loss that enforces astrophysical parameter consistency
- **Multi-Attribute Conditioning**: Simultaneous control over morphology (4 classes) and physical parameters (4 continuous attributes)
- **Rare Class Generation**: Capability to generate underrepresented morphologies (mergers, edge-on galaxies)
- **Interactive Web Interface**: Real-time galaxy generation with parameter controls
- **Scientific Validation**: Physics loss < 0.01 demonstrates strong attribute matching

### Morphology Classes

1. **Spiral Galaxies** - Disk galaxies with spiral arm structures
2. **Elliptical Galaxies** - Smooth, featureless elliptical morphologies
3. **Merger Galaxies** - Interacting or merging galaxy systems
4. **Edge-on Galaxies** - Disk galaxies viewed from the side

### Physical Parameters

- **Size**: Galaxy angular size (0.3 - 1.0)
- **Brightness**: Apparent brightness (0.2 - 1.0)
- **Ellipticity**: Shape ellipticity (0.0 - 0.9)
- **Redshift**: Cosmological redshift (0.0 - 0.5)

## Architecture

### Generator Network
- Input: 100D noise vector + 8D condition vector (4 morphology one-hot + 4 physical attributes)
- Architecture: DCGAN with 4 transposed convolutional layers
- Output: 64×64 RGB galaxy image
- Parameters: ~2.5M

### Discriminator Network
- Input: 64×64 image + 8D condition (spatially replicated)
- Architecture: 4 convolutional layers with BatchNorm and LeakyReLU
- Output: Real/fake probability
- Parameters: ~1.8M

### Physics Predictor Network
- Purpose: Ensures generated galaxies match requested physical attributes
- Input: 64×64 galaxy image
- Architecture: 3 convolutional layers + 2 fully connected layers
- Output: 4D vector (predicted physical attributes)
- Parameters: ~450K

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (RTX 3070 or better recommended)
- 5GB disk space for checkpoints
- 2GB VRAM during training

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/physics-aware-galaxy-gan.git
cd physics-aware-galaxy-gan

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the **Galaxy Zoo 2** dataset (Hart et al. 2016):

- **Size**: ~300,000 galaxy images with morphology classifications
- **Source**: [Galaxy Zoo 2 on Zenodo](https://zenodo.org/record/3565489)
- **Labels**: [data.galaxyzoo.org](https://data.galaxyzoo.org/)

### Data Structure

```
data/
├── gz2_hart16.csv
└── images_gz2/
    └── images/
        ├── 587722981862.jpg
        ├── 587722981863.jpg
        └── ...
```

## Usage

### Training

#### Train GAN (Default)
```bash
python galaxy.py
```

#### Train Diffusion Model (Alternative)
```bash
python galaxy.py --diffusion
```

#### Resume Training from Checkpoint
```bash
python resume_training.py
```

### Generation

#### Generate from Latest Checkpoint
```bash
python generate_from_checkpoint.py
```

#### Run Interactive Web Interface
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

### Utilities

#### Check Dataset
```bash
python check_data.py
```

#### Test Model Architecture
```bash
python test_models.py
```

#### Debug Image Loading
```bash
python debug_images.py
```

## Training Configuration

### Optimized Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 64 | Stable gradients, better generalization |
| Learning Rate | 0.00015 | Slower, more stable training |
| Epochs | 150 | Sufficient for convergence |
| Physics Weight (λ) | 3.0 | Balanced realism + attribute control |
| Optimizer | Adam (β₁=0.5, β₂=0.999) | Standard for GANs |
| Gradient Clipping | 1.0 | Prevents exploding gradients |

### Training Time (RTX 3070)
- Physics pre-training: ~25 minutes
- GAN training: ~4 hours
- **Total: ~4.5 hours**

## Loss Functions

### Discriminator Loss
```
L_D = BCE(D(real, c), 0.9) + BCE(D(G(z, c), c), 0.1)
```
*Note: Label smoothing (0.9/0.1) for stability*

### Generator Loss (Physics-Aware)
```
L_G = L_GAN + λ_physics × L_physics

where:
L_GAN = BCE(D(G(z, c), c), 1.0)
L_physics = MSE(PhysicsPredictor(G(z, c)), c_physical)
λ_physics = 3.0
```

## Results

### Training Metrics (Epoch 150)
- Discriminator Loss: ~0.68
- Generator Loss: ~0.72
- Physics Loss: ~0.006 (excellent attribute matching)
- Training Stability: No mode collapse observed

### Visual Quality
- Galaxy-like circular/elliptical shapes
- Distinguishable spiral vs elliptical morphologies
- Variation in size and brightness
- Realistic structural features

## Project Structure

```
final-project-files/
├── galaxy.py                    # Main training script (GAN & Diffusion)
├── gan_model.py                 # GAN architecture
├── physics_model.py             # Physics predictor network
├── diffusion_model.py           # Diffusion model architecture
├── cgan_physics.py              # Physics-aware CGAN implementation
├── config.py                    # Basic configuration
├── config_physics.py            # Physics-aware configuration
├── resume_training.py           # Resume training from checkpoint
├── generate_from_checkpoint.py  # Generation script
├── app.py                       # Flask web server
├── frontend/                    # Web interface
│   ├── index.html              # UI
│   ├── style.css               # Styling
│   └── script.js               # Interactivity
├── requirements.txt             # Dependencies
├── checkpoints/                 # Model checkpoints
├── samples/                     # Generated samples
├── data/                        # Galaxy Zoo 2 data
├── README.md                    # This file
├── PROJECT_REPORT.md            # Detailed project report
├── DATA_USAGE.md                # Dataset documentation
└── runcommands.md               # Quick command reference
```

## Configuration Files

### `config.py` - Basic Configuration
- Image size, batch size, epochs
- Learning rate, noise dimension
- Device settings

### `config_physics.py` - Physics-Aware Configuration
- Physics predictor settings
- Physics loss weight
- Pre-training epochs
- Extended training parameters

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` to 32 or 16
- Ensure no other GPU processes running
- Lower image resolution if needed

### Images Look Noisy
- Train for more epochs (100-150)
- Check data loading with `debug_images.py`
- Verify data normalization

### Mode Collapse
- Reduce learning rate to 0.0001
- Increase discriminator training frequency
- Add label smoothing (already implemented)

### High Physics Loss
- Increase pre-training epochs
- Reduce physics loss weight (λ)
- Check physics predictor convergence

## Scientific Validation

### Evaluation Metrics

| Metric | Target Range | Interpretation |
|--------|--------------|----------------|
| D Loss | 0.6 - 0.7 | Balanced discriminator |
| G Loss | 0.7 - 1.5 | Generator improving |
| Physics Loss | < 0.01 | Good attribute matching |
| D_real | 0.85 - 0.90 | Recognizes real images |
| D_fake | 0.10 - 0.15 | Detects fake images |

## Why GAN Over Diffusion?

While diffusion models represent state-of-the-art in image generation, our GAN-based approach was more appropriate for this project:

| Aspect | GAN (Our Choice) | Diffusion |
|--------|------------------|-----------|
| Training Time | **4.5 hours** | Weeks |
| Inference Speed | **Fast (single pass)** | Slow (iterative) |
| Physics Integration | **Natural via auxiliary predictor** | Complex |
| Resource Efficiency | **Single RTX 3070** | Multiple GPUs |
| Interpretability | **Clear adversarial objective** | Black box |

## Future Work

### Short-term Improvements
- Higher resolution (128×128 or 256×256)
- More physical parameters (stellar mass, color, concentration)
- Real redshift data integration
- Perceptual loss for better quality

### Long-term Extensions
- Classification augmentation for rare types
- Multi-wavelength generation (UV, optical, IR)
- 3D structure incorporation
- Spectral data conditioning

## Research Applications

- **Data Augmentation**: Improve classification of rare galaxy types
- **Simulation Validation**: Compare with cosmological simulations
- **Survey Planning**: Generate realistic mock observations
- **Education**: Interactive tool for teaching galaxy morphology

## References

1. Hart, R. E., et al. (2016). Galaxy Zoo: comparing the demographics of spiral arm number and a new method for correcting redshift bias. *Monthly Notices of the Royal Astronomical Society*, 461(4), 3663-3682.

2. Lanusse, F., et al. (2021). Deep generative models for galaxy image simulation. *Astronomy & Astrophysics*, 646, A13.

3. Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. *arXiv preprint arXiv:1511.06434*.

4. Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*.

## Documentation

For more detailed information, see:
- [PROJECT_REPORT.md](PROJECT_REPORT.md) - Comprehensive project report
- [DATA_USAGE.md](DATA_USAGE.md) - Dataset documentation
- [runcommands.md](runcommands.md) - Quick command reference

## License

This project is for educational and research purposes. The Galaxy Zoo 2 dataset is used under its original license terms.

## Acknowledgments

- Galaxy Zoo 2 team for providing the dataset
- Hart et al. (2016) for the morphology classifications
- PyTorch team for the deep learning framework

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{physics-aware-galaxy-gan,
  title={Physics-Aware Galaxy Generation with Conditional GANs},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/physics-aware-galaxy-gan}
}
```

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
