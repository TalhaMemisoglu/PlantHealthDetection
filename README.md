# Plant Health Detection using Digital Image Processing

A digital image processing project that detects whether plant leaves are healthy or unhealthy using **traditional DIP techniques only** - no AI/ML involved.

## Features

This system uses the following DIP techniques to analyze plant leaf images:

### 1. Color Analysis (HSV Color Space)
- **Green Ratio**: Measures the proportion of green pixels (healthy leaves are greener)
- **Brown/Yellow Ratio**: Detects discoloration indicating disease
- **Excess Green Index**: Vegetation index for greenness measurement
- **Hue Standard Deviation**: Measures color variation (higher = more disease)

### 2. Texture Analysis (GLCM)
- **Contrast**: Measures local intensity variations
- **Homogeneity**: Measures uniformity of texture
- **Energy**: Measures texture uniformity
- **Entropy**: Measures randomness in texture

### 3. Spot Detection
- **Brown Spots**: Detects bacterial spot, early/late blight
- **Yellow Spots**: Detects leaf mold, septoria
- **Dark Spots**: Detects necrotic tissue
- **Light Spots**: Detects spider mite damage

### 4. Edge Analysis
- **Edge Density**: Canny edge detection within leaf region
- **Gradient Analysis**: Sobel operators for texture analysis

### 5. Histogram Analysis
- **Color Distribution**: RGB histogram statistics
- **Green Hue Concentration**: Peak analysis in green spectrum

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- OpenCV (cv2) - Image processing
- NumPy - Numerical operations
- scikit-image - GLCM texture features
- SciPy - Scientific computing
- Matplotlib - Visualization

## Usage

### Interactive Mode

```bash
python main.py
```

This opens an interactive menu with options:
1. Analyze a single image
2. Evaluate on sample dataset (20 images)
3. Evaluate on full dataset
4. Compare healthy vs unhealthy samples
5. Exit

### Programmatic Usage

```python
from main import PlantHealthDetector

# Initialize detector
detector = PlantHealthDetector()

# Analyze single image
analysis = detector.analyze_image("path/to/leaf.jpg")

# Get classification result
print(f"Classification: {analysis['result']['classification']}")
print(f"Health Score: {analysis['result']['health_score']}/100")
print(f"Is Healthy: {analysis['result']['is_healthy']}")

# Visualize results
detector.visualize_analysis(analysis)
```

### Batch Evaluation

```python
from main import PlantHealthDetector, evaluate_dataset

detector = PlantHealthDetector()

# Evaluate on datasets
results = evaluate_dataset(
    detector,
    healthy_dir="./healthy",
    unhealthy_dir="./unhealthy",
    sample_size=50  # or None for full dataset
)
```

## Classification Algorithm

The system uses a **weighted scoring approach**:

1. Start with 100 health points
2. Subtract penalties based on disease indicators:
   - Low green ratio → -25 points max
   - High brown/yellow ratio → -25 points max
   - Brown spots detected → -15 points max
   - Disease area ratio → -15 points max
   - High color variation → -10 points max
   - Low green intensity → -10 points max

3. Final classification:
   - **HEALTHY**: Score ≥ 75
   - **UNHEALTHY (Mild)**: Score 50-74
   - **UNHEALTHY (Severe)**: Score < 50

## Performance

On the provided dataset (78 healthy + 205 unhealthy images):

| Metric | Accuracy |
|--------|----------|
| Healthy Detection | ~90% |
| Unhealthy Detection | ~66% |
| **Overall Accuracy** | **~72%** |

*Note: This accuracy is achieved without any machine learning, using purely rule-based DIP techniques.*

## Project Structure

```
DIP/
├── main.py           # Main detection system
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── healthy/          # Healthy leaf images (78 samples)
└── unhealthy/        # Unhealthy leaf images (205 samples)
```

## How It Works

### Image Processing Pipeline

1. **Load & Resize**: Image loaded and resized to 256x256
2. **Denoise**: Bilateral filter to reduce noise while preserving edges
3. **Segmentation**: Color-based leaf segmentation from background
4. **Feature Extraction**: Extract color, texture, edge, and spot features
5. **Classification**: Rule-based scoring using thresholds

### Key DIP Techniques Used

| Technique | Purpose |
|-----------|---------|
| HSV Color Space | Color analysis independent of lighting |
| Bilateral Filtering | Noise reduction preserving edges |
| Morphological Operations | Mask cleanup (open/close) |
| Canny Edge Detection | Detect edges within leaf |
| Sobel Operators | Gradient/texture analysis |
| GLCM | Texture feature extraction |
| Connected Components | Count disease spots |
| Color Histograms | Distribution analysis |

## Limitations

1. **Subtle diseases**: Early-stage diseases with minimal visual symptoms may be missed
2. **Lighting variations**: Extreme lighting may affect color-based features
3. **Background noise**: Complex backgrounds may interfere with segmentation
4. **Disease types**: Different disease types require different color ranges

## Future Improvements (without ML)

1. Add adaptive thresholding based on image statistics
2. Implement multi-scale analysis for different spot sizes
3. Add shape-based features for leaf structure analysis
4. Implement LAB color space analysis for lighting invariance
5. Add Gabor filter for better texture analysis

## License

This project is for educational purposes as part of a Digital Image Processing course.

