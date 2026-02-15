// State Management
let currentMorphology = 0;
const morphologyNames = ['Spiral', 'Elliptical', 'Merger', 'Edge-on'];

// DOM Elements
const morphButtons = document.querySelectorAll('.morph-btn');
const generateBtn = document.getElementById('generate-btn');
const galaxyDisplay = document.getElementById('galaxy-display');
const galaxyImage = document.getElementById('galaxy-image');
const loadingSpinner = document.getElementById('loading-spinner');
const infoDisplay = document.getElementById('info-display');

// Sliders
const sizeSlider = document.getElementById('size-slider');
const brightnessSlider = document.getElementById('brightness-slider');
const ellipticitySlider = document.getElementById('ellipticity-slider');
const redshiftSlider = document.getElementById('redshift-slider');

// Value Displays
const sizeValue = document.getElementById('size-value');
const brightnessValue = document.getElementById('brightness-value');
const ellipticityValue = document.getElementById('ellipticity-value');
const redshiftValue = document.getElementById('redshift-value');

// Info Display Elements
const infoMorphology = document.getElementById('info-morphology');
const infoSize = document.getElementById('info-size');
const infoBrightness = document.getElementById('info-brightness');
const infoEllipticity = document.getElementById('info-ellipticity');
const infoRedshift = document.getElementById('info-redshift');

// Event Listeners - Morphology Buttons
morphButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons
        morphButtons.forEach(b => b.classList.remove('active'));
        
        // Add active class to clicked button
        btn.classList.add('active');
        
        // Update current morphology
        currentMorphology = parseInt(btn.dataset.class);
        
        // Add click animation
        btn.style.transform = 'scale(0.95)';
        setTimeout(() => {
            btn.style.transform = '';
        }, 150);
    });
});

// Event Listeners - Sliders
sizeSlider.addEventListener('input', (e) => {
    sizeValue.textContent = parseFloat(e.target.value).toFixed(2);
});

brightnessSlider.addEventListener('input', (e) => {
    brightnessValue.textContent = parseFloat(e.target.value).toFixed(2);
});

ellipticitySlider.addEventListener('input', (e) => {
    ellipticityValue.textContent = parseFloat(e.target.value).toFixed(2);
});

redshiftSlider.addEventListener('input', (e) => {
    redshiftValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// Generate Button Click Handler
generateBtn.addEventListener('click', async () => {
    // Disable button during generation
    generateBtn.disabled = true;
    generateBtn.style.opacity = '0.6';
    generateBtn.style.cursor = 'not-allowed';
    
    // Hide previous image and info
    galaxyImage.style.display = 'none';
    infoDisplay.style.display = 'none';
    
    // Show loading spinner
    const placeholder = document.querySelector('.placeholder-content');
    if (placeholder) placeholder.style.display = 'none';
    loadingSpinner.style.display = 'block';
    
    // Collect parameters
    const params = {
        morphology: currentMorphology,
        size: parseFloat(sizeSlider.value),
        brightness: parseFloat(brightnessSlider.value),
        ellipticity: parseFloat(ellipticitySlider.value),
        redshift: parseFloat(redshiftSlider.value)
    };
    
    try {
        // Call API
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Display generated image
            galaxyImage.src = data.image;
            galaxyImage.style.display = 'block';
            
            // Update info display
            infoMorphology.textContent = morphologyNames[currentMorphology];
            infoSize.textContent = params.size.toFixed(2);
            infoBrightness.textContent = params.brightness.toFixed(2);
            infoEllipticity.textContent = params.ellipticity.toFixed(2);
            infoRedshift.textContent = params.redshift.toFixed(2);
            infoDisplay.style.display = 'grid';
            
        } else {
            throw new Error(data.error || 'Generation failed');
        }
        
    } catch (error) {
        console.error('Error generating galaxy:', error);
        
        // Show error message
        loadingSpinner.innerHTML = `
            <div style="color: #f5576c; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <p style="font-size: 1.1rem; font-weight: 500;">Generation Failed</p>
                <p style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.5); margin-top: 0.5rem;">
                    ${error.message}
                </p>
                <p style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.4); margin-top: 0.5rem;">
                    Make sure the server is running
                </p>
            </div>
        `;
        
        // Reset loading spinner after 3 seconds
        setTimeout(() => {
            loadingSpinner.innerHTML = `
                <div class="spinner"></div>
                <p class="loading-text">Generating galaxy...</p>
            `;
        }, 3000);
        
    } finally {
        // Hide loading spinner
        loadingSpinner.style.display = 'none';
        
        // Re-enable button
        generateBtn.disabled = false;
        generateBtn.style.opacity = '1';
        generateBtn.style.cursor = 'pointer';
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Press Enter to generate
    if (e.key === 'Enter' && !generateBtn.disabled) {
        generateBtn.click();
    }
    
    // Press 1-4 to select morphology
    if (e.key >= '1' && e.key <= '4') {
        const index = parseInt(e.key) - 1;
        morphButtons[index].click();
    }
});

// Add smooth scroll behavior
document.documentElement.style.scrollBehavior = 'smooth';

// Console welcome message
console.log('%c🌌 Galaxy CGAN Generator', 'font-size: 20px; font-weight: bold; color: #667eea;');
console.log('%cPhysics-Aware Conditional GAN for Galaxy Generation', 'font-size: 12px; color: #999;');
console.log('%cKeyboard Shortcuts:', 'font-size: 14px; font-weight: bold; margin-top: 10px;');
console.log('  • Enter: Generate Galaxy');
console.log('  • 1-4: Select Morphology Class');
