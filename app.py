"""
Flask Web Server for Galaxy CGAN
Serves the frontend and provides API endpoint for galaxy generation
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.utils as vutils
import base64
import io
from PIL import Image
import os

from config import *
from gan_model import Generator

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for development


# Generator imported from gan_model


# Global variables for model
generator = None
device = DEVICE


def load_model():
    """Load the generator model from checkpoint"""
    global generator
    
    # Find the latest checkpoint
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in checkpoints directory")
    
    # Sort by epoch number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    # Initialize and load generator
    generator = Generator(NOISE_DIM, CONDITION_DIM, IMAGE_SIZE).to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    epoch = checkpoint['epoch'] + 1
    print(f"✓ Model loaded successfully from epoch {epoch}")
    print(f"✓ Device: {device}")


def tensor_to_base64(tensor):
    """Convert a tensor image to base64 string"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    tensor = tensor.cpu()
    img_array = tensor.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype('uint8')
    img = Image.fromarray(img_array)
    
    # Resize for better display (upscale from 64x64 to 256x256)
    img = img.resize((256, 256), Image.LANCZOS)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('frontend', 'index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a galaxy image based on parameters"""
    try:
        # Get parameters from request
        data = request.json
        morphology = int(data.get('morphology', 0))
        size = float(data.get('size', 0.65))
        brightness = float(data.get('brightness', 0.60))
        ellipticity = float(data.get('ellipticity', 0.45))
        redshift = float(data.get('redshift', 0.25))
        
        # Validate parameters
        if morphology not in [0, 1, 2, 3]:
            return jsonify({'success': False, 'error': 'Invalid morphology class'}), 400
        
        if not (0.3 <= size <= 1.0):
            return jsonify({'success': False, 'error': 'Size must be between 0.3 and 1.0'}), 400
        
        if not (0.2 <= brightness <= 1.0):
            return jsonify({'success': False, 'error': 'Brightness must be between 0.2 and 1.0'}), 400
        
        if not (0.0 <= ellipticity <= 0.9):
            return jsonify({'success': False, 'error': 'Ellipticity must be between 0.0 and 0.9'}), 400
        
        if not (0.0 <= redshift <= 0.5):
            return jsonify({'success': False, 'error': 'Redshift must be between 0.0 and 0.5'}), 400
        
        # Create condition vector
        morph_onehot = torch.zeros(4)
        morph_onehot[morphology] = 1.0
        
        condition = torch.cat([
            morph_onehot,
            torch.tensor([size, brightness, ellipticity, redshift])
        ]).unsqueeze(0).to(device)
        
        # Generate random noise
        noise = torch.randn(1, NOISE_DIM, device=device)
        
        # Generate image
        with torch.no_grad():
            fake_image = generator(noise, condition)
        
        # Convert to base64
        image_base64 = tensor_to_base64(fake_image[0])
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'parameters': {
                'morphology': morphology,
                'size': size,
                'brightness': brightness,
                'ellipticity': ellipticity,
                'redshift': redshift
            }
        })
        
    except Exception as e:
        print(f"Error generating galaxy: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': generator is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🌌 Galaxy CGAN Web Server")
    print("=" * 60)
    
    # Load the model
    try:
        load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have trained checkpoints in the 'checkpoints' directory")
        exit(1)
    
    print("\n" + "=" * 60)
    print("🚀 Starting web server...")
    print("=" * 60)
    print(f"📍 Server running at: http://localhost:5000")
    print(f"🌐 Open your browser and navigate to the URL above")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
