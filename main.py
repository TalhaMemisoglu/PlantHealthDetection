"""
Digital Image Processing Project: Plant Health Detection
This project uses traditional DIP techniques (no AI/ML) to classify plant leaves as healthy or unhealthy.

Features extracted:
1. Color Features (HSV analysis) - Green ratio, brown/yellow detection
2. Texture Features (GLCM) - Contrast, homogeneity, entropy, energy
3. Edge Features - Edge density for spot detection
4. Spot Detection - Brown/yellow spot area ratio
5. Histogram Features - Color distribution analysis
"""

import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
import matplotlib.pyplot as plt
from collections import defaultdict


class PlantHealthDetector:
    """
    Plant health detector using Digital Image Processing techniques.
    No machine learning - uses rule-based classification with extracted features.
    """
    
    def __init__(self):
        # Thresholds optimized via grid search on balanced dataset (78 healthy, 78 unhealthy)
        # Achieves ~81% overall accuracy with balanced healthy/unhealthy detection
        self.thresholds = {
            'green_ratio_min': 0.90,           # Healthy: ~0.96, Unhealthy: ~0.69
            'brown_spot_ratio_max': 0.02,      # Healthy: ~0.007, Unhealthy: ~0.107
            'yellow_ratio_max': 0.08,          # Healthy: ~0.042, Unhealthy: ~0.275
            'disease_ratio_max': 0.12,         # Healthy: ~0.10, Unhealthy: ~0.37
            'excess_green_min': 65,            # Healthy: ~66, Unhealthy: ~50
            'hue_std_max': 7.5,                # Healthy: ~6.5, Unhealthy: ~11.5
        }
        self.health_score_threshold = 85      # Classification threshold
        
    def load_image(self, image_path):
        """Load and validate image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def preprocess_image(self, img):
        """
        Preprocess image: resize, denoise, and create mask for leaf region.
        """
        # Resize for consistent processing
        img_resized = cv2.resize(img, (256, 256))
        
        # Denoise using bilateral filter (preserves edges)
        img_denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
        
        # Create leaf mask (segment leaf from background)
        leaf_mask = self._create_leaf_mask(img_denoised)
        
        return img_resized, img_denoised, leaf_mask
    
    def _create_leaf_mask(self, img):
        """
        Create binary mask to isolate leaf from background.
        Uses color-based segmentation in HSV space.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for green colors (leaf region)
        # Also include yellow-brown for diseased areas
        lower_green = np.array([20, 20, 20])
        upper_green = np.array([100, 255, 255])
        
        # Create mask for green region
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Include brown/yellow diseased areas
        lower_brown = np.array([5, 30, 30])
        upper_brown = np.array([25, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_green, mask_brown)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only the largest contour (main leaf)
            largest_contour = max(contours, key=cv2.contourArea)
            filled_mask = np.zeros_like(combined_mask)
            cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)
            return filled_mask
        
        return combined_mask
    
    # ==================== FEATURE EXTRACTION ====================
    
    def extract_color_features(self, img, mask):
        """
        Extract color-based features in HSV and LAB color spaces.
        Key insight: Healthy leaves have more uniform green color.
        """
        features = {}
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Apply mask to get only leaf pixels
        leaf_pixels_hsv = hsv[mask > 0]
        leaf_pixels_bgr = img[mask > 0]
        
        if len(leaf_pixels_hsv) == 0:
            return self._default_color_features()
        
        # Extract HSV statistics
        h_channel = leaf_pixels_hsv[:, 0]
        s_channel = leaf_pixels_hsv[:, 1]
        v_channel = leaf_pixels_hsv[:, 2]
        
        features['hue_mean'] = np.mean(h_channel)
        features['hue_std'] = np.std(h_channel)
        features['saturation_mean'] = np.mean(s_channel)
        features['saturation_std'] = np.std(s_channel)
        features['value_mean'] = np.mean(v_channel)
        features['value_std'] = np.std(v_channel)
        
        # Color uniformity - diseased leaves have higher hue/saturation variance
        features['hue_variance'] = np.var(h_channel)
        features['saturation_variance'] = np.var(s_channel)
        
        # Calculate green ratio (healthy indicator)
        # Green hue range: approximately 35-85 in OpenCV HSV
        green_mask = (h_channel >= 35) & (h_channel <= 85)
        features['green_ratio'] = np.sum(green_mask) / len(h_channel)
        
        # Calculate brown/yellow ratio (disease indicator)
        # Brown/yellow hue range: approximately 10-30
        brown_yellow_mask = (h_channel >= 10) & (h_channel <= 35)
        features['brown_yellow_ratio'] = np.sum(brown_yellow_mask) / len(h_channel)
        
        # Calculate dark spots ratio (potential disease)
        dark_mask = v_channel < 80
        features['dark_ratio'] = np.sum(dark_mask) / len(v_channel)
        
        # RGB channel analysis
        b_channel = leaf_pixels_bgr[:, 0]
        g_channel = leaf_pixels_bgr[:, 1]
        r_channel = leaf_pixels_bgr[:, 2]
        
        # Green dominance ratio
        total_rgb = b_channel.astype(float) + g_channel.astype(float) + r_channel.astype(float) + 1
        features['green_dominance'] = np.mean(g_channel.astype(float) / total_rgb)
        features['red_ratio'] = np.mean(r_channel.astype(float) / total_rgb)
        
        # Excess Green Index (ExG) - common in vegetation analysis
        features['excess_green'] = np.mean(2 * g_channel.astype(float) - r_channel.astype(float) - b_channel.astype(float))
        
        return features
    
    def _default_color_features(self):
        """Return default color features when mask is empty."""
        return {
            'hue_mean': 0, 'hue_std': 0,
            'saturation_mean': 0, 'saturation_std': 0,
            'value_mean': 0, 'value_std': 0,
            'hue_variance': 0, 'saturation_variance': 0,
            'green_ratio': 0, 'brown_yellow_ratio': 1,
            'dark_ratio': 0, 'green_dominance': 0,
            'red_ratio': 0, 'excess_green': 0
        }
    
    def extract_texture_features(self, img, mask):
        """
        Extract texture features using GLCM (Gray Level Co-occurrence Matrix).
        Diseased areas typically show different texture patterns.
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Reduce gray levels for GLCM computation (256 -> 64)
        gray_reduced = (gray_masked // 4).astype(np.uint8)
        
        # Compute GLCM for multiple angles
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(gray_reduced, distances=distances, angles=angles,
                               levels=64, symmetric=True, normed=True)
            
            # Extract GLCM properties
            features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            
        except Exception as e:
            # Default values if GLCM fails
            features['contrast'] = 0
            features['homogeneity'] = 1
            features['energy'] = 1
            features['correlation'] = 1
            features['dissimilarity'] = 0
        
        # Calculate entropy from histogram
        hist = cv2.calcHist([gray_masked], [0], mask, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        features['entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return features
    
    def extract_edge_features(self, img, mask):
        """
        Extract edge-based features.
        Disease spots often create additional edges within the leaf.
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(gray_blurred, 50, 150)
        
        # Apply mask to consider only leaf region
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Calculate edge density (edges per leaf pixel)
        leaf_area = np.sum(mask > 0)
        edge_pixels = np.sum(edges_masked > 0)
        
        features['edge_density'] = edge_pixels / (leaf_area + 1) if leaf_area > 0 else 0
        
        # Sobel gradients for texture analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_masked = gradient_magnitude * (mask > 0)
        
        features['gradient_mean'] = np.mean(gradient_masked[mask > 0]) if np.sum(mask > 0) > 0 else 0
        features['gradient_std'] = np.std(gradient_masked[mask > 0]) if np.sum(mask > 0) > 0 else 0
        
        return features
    
    def detect_disease_spots(self, img, mask):
        """
        Detect brown/yellow disease spots using color segmentation.
        Returns the ratio of diseased area to total leaf area.
        """
        features = {}
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for disease spots (brown, yellow, dark spots)
        
        # Brown spots (Early Blight, Late Blight)
        lower_brown = np.array([8, 50, 30])
        upper_brown = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Yellow spots (various diseases) - extended range
        lower_yellow = np.array([18, 40, 80])
        upper_yellow = np.array([38, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Light spots (Spider mites, early damage) - whitish/pale areas
        lower_light = np.array([0, 0, 180])
        upper_light = np.array([180, 50, 255])
        light_mask = cv2.inRange(hsv, lower_light, upper_light)
        
        # Dark necrotic spots
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Combine disease masks
        disease_mask = cv2.bitwise_or(brown_mask, yellow_mask)
        disease_mask = cv2.bitwise_or(disease_mask, dark_mask)
        disease_mask = cv2.bitwise_or(disease_mask, light_mask)
        
        # Apply leaf mask
        disease_in_leaf = cv2.bitwise_and(disease_mask, disease_mask, mask=mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        disease_in_leaf = cv2.morphologyEx(disease_in_leaf, cv2.MORPH_OPEN, kernel)
        
        # Calculate ratios
        leaf_area = np.sum(mask > 0)
        brown_area = np.sum(cv2.bitwise_and(brown_mask, mask) > 0)
        yellow_area = np.sum(cv2.bitwise_and(yellow_mask, mask) > 0)
        dark_area = np.sum(cv2.bitwise_and(dark_mask, mask) > 0)
        disease_area = np.sum(disease_in_leaf > 0)
        
        light_area = np.sum(cv2.bitwise_and(light_mask, mask) > 0)
        
        features['brown_spot_ratio'] = brown_area / (leaf_area + 1) if leaf_area > 0 else 0
        features['yellow_spot_ratio'] = yellow_area / (leaf_area + 1) if leaf_area > 0 else 0
        features['dark_spot_ratio'] = dark_area / (leaf_area + 1) if leaf_area > 0 else 0
        features['light_spot_ratio'] = light_area / (leaf_area + 1) if leaf_area > 0 else 0
        features['total_disease_ratio'] = disease_area / (leaf_area + 1) if leaf_area > 0 else 0
        
        # Count number of distinct spots using connected components
        num_labels, labels = cv2.connectedComponents(disease_in_leaf)
        features['num_disease_spots'] = max(0, num_labels - 1)  # Subtract background
        
        # Store mask for visualization
        features['disease_mask'] = disease_in_leaf
        
        return features
    
    def extract_histogram_features(self, img, mask):
        """
        Extract histogram-based features for color distribution analysis.
        """
        features = {}
        
        # Calculate histograms for each channel
        colors = ('b', 'g', 'r')
        hist_features = []
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], mask, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-10)  # Normalize
            
            # Statistical features from histogram
            features[f'{color}_hist_mean'] = np.mean(hist)
            features[f'{color}_hist_std'] = np.std(hist)
            features[f'{color}_hist_max'] = np.max(hist)
            features[f'{color}_hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Color uniformity (lower for diseased leaves)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], mask, [30], [0, 180])
        h_hist = h_hist.flatten()
        h_hist = h_hist / (h_hist.sum() + 1e-10)
        
        # Concentration of green hues (hue around 35-85)
        green_hue_bins = h_hist[5:15]  # Approximate green range
        features['green_hue_concentration'] = np.sum(green_hue_bins)
        
        return features
    
    def extract_all_features(self, img, mask):
        """
        Extract all features from an image.
        """
        all_features = {}
        
        # Extract feature groups
        color_features = self.extract_color_features(img, mask)
        texture_features = self.extract_texture_features(img, mask)
        edge_features = self.extract_edge_features(img, mask)
        spot_features = self.detect_disease_spots(img, mask)
        histogram_features = self.extract_histogram_features(img, mask)
        
        # Combine all features (exclude mask from spot features)
        all_features.update(color_features)
        all_features.update(texture_features)
        all_features.update(edge_features)
        all_features.update({k: v for k, v in spot_features.items() if k != 'disease_mask'})
        all_features.update(histogram_features)
        
        # Store disease mask separately
        all_features['_disease_mask'] = spot_features.get('disease_mask', None)
        
        return all_features
    
    # ==================== CLASSIFICATION ====================
    
    def classify_health(self, features):
        """
        Classify plant health based on extracted features.
        Uses rule-based classification with weighted scoring.
        """
        health_score = 100  # Start with perfect health
        reasons = []
        
        # Feature values
        green_ratio = features.get('green_ratio', 0)
        brown_yellow_ratio = features.get('brown_yellow_ratio', 0)
        brown_spot_ratio = features.get('brown_spot_ratio', 0)
        total_disease_ratio = features.get('total_disease_ratio', 0)
        excess_green = features.get('excess_green', 0)
        hue_std = features.get('hue_std', 0)
        light_spot_ratio = features.get('light_spot_ratio', 0)
        
        # 1. Green ratio check (weight: 25 points max)
        # Healthy ~0.95, Unhealthy ~0.76
        if green_ratio < self.thresholds['green_ratio_min']:
            deficit = self.thresholds['green_ratio_min'] - green_ratio
            penalty = min(25, deficit * 130)
            health_score -= penalty
            reasons.append(f"Reduced green area: {green_ratio:.1%}")
        
        # 2. Brown/Yellow ratio check (weight: 25 points max)
        # Healthy ~0.045, Unhealthy ~0.213
        if brown_yellow_ratio > self.thresholds['yellow_ratio_max']:
            excess = brown_yellow_ratio - self.thresholds['yellow_ratio_max']
            penalty = min(25, excess * 180)
            health_score -= penalty
            reasons.append(f"Yellow/brown discoloration: {brown_yellow_ratio:.1%}")
        
        # 3. Brown spot ratio check (weight: 15 points max)
        # Healthy ~0.008, Unhealthy ~0.077
        if brown_spot_ratio > self.thresholds['brown_spot_ratio_max']:
            excess = brown_spot_ratio - self.thresholds['brown_spot_ratio_max']
            penalty = min(15, excess * 250)
            health_score -= penalty
            reasons.append(f"Brown spots detected: {brown_spot_ratio:.1%}")
        
        # 4. Total disease area check (weight: 15 points max)
        # Healthy ~0.08, Unhealthy ~0.20
        if total_disease_ratio > self.thresholds['disease_ratio_max']:
            excess = total_disease_ratio - self.thresholds['disease_ratio_max']
            penalty = min(15, excess * 130)
            health_score -= penalty
            reasons.append(f"Disease area: {total_disease_ratio:.1%}")
        
        # 5. Hue standard deviation check (weight: 10 points max)
        # Higher color variation indicates disease
        if hue_std > self.thresholds['hue_std_max']:
            excess = hue_std - self.thresholds['hue_std_max']
            penalty = min(10, excess * 0.8)
            health_score -= penalty
            reasons.append(f"Color variation: {hue_std:.1f}")
        
        # 6. Excess green index check (weight: 10 points max)
        # Healthy ~64, Unhealthy ~51
        if excess_green < self.thresholds['excess_green_min']:
            deficit = self.thresholds['excess_green_min'] - excess_green
            penalty = min(10, deficit * 0.6)
            health_score -= penalty
            reasons.append(f"Low green intensity: {excess_green:.1f}")
        
        # Ensure score is within bounds
        health_score = max(0, min(100, health_score))
        
        # Classification using optimized threshold
        threshold = self.health_score_threshold
        if health_score >= threshold:
            classification = "HEALTHY"
        elif health_score >= threshold - 25:
            classification = "UNHEALTHY (Mild)"
        else:
            classification = "UNHEALTHY (Severe)"
        
        return {
            'classification': classification,
            'health_score': health_score,
            'is_healthy': health_score >= threshold,
            'reasons': reasons if reasons else ["Leaf appears healthy"]
        }
    
    def analyze_image(self, image_path):
        """
        Complete analysis pipeline for a single image.
        """
        # Load image
        img = self.load_image(image_path)
        
        # Preprocess
        img_resized, img_denoised, leaf_mask = self.preprocess_image(img)
        
        # Extract features
        features = self.extract_all_features(img_denoised, leaf_mask)
        
        # Classify
        result = self.classify_health(features)
        
        return {
            'image_path': image_path,
            'features': features,
            'result': result,
            'processed_images': {
                'original': img_resized,
                'denoised': img_denoised,
                'leaf_mask': leaf_mask,
                'disease_mask': features.get('_disease_mask')
            }
        }
    
    def visualize_analysis(self, analysis, save_path=None):
        """
        Visualize the analysis results with multiple views.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        original_rgb = cv2.cvtColor(analysis['processed_images']['original'], cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Leaf mask
        axes[0, 1].imshow(analysis['processed_images']['leaf_mask'], cmap='gray')
        axes[0, 1].set_title('Leaf Segmentation')
        axes[0, 1].axis('off')
        
        # Disease spots highlighted
        disease_mask = analysis['processed_images']['disease_mask']
        if disease_mask is not None:
            overlay = original_rgb.copy()
            overlay[disease_mask > 0] = [255, 0, 0]  # Red overlay for disease
            blended = cv2.addWeighted(original_rgb, 0.7, overlay, 0.3, 0)
            axes[0, 2].imshow(blended)
        else:
            axes[0, 2].imshow(original_rgb)
        axes[0, 2].set_title('Disease Spots (Red)')
        axes[0, 2].axis('off')
        
        # Color histogram
        colors = ['blue', 'green', 'red']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([analysis['processed_images']['denoised']], [i], 
                               analysis['processed_images']['leaf_mask'], [256], [0, 256])
            axes[1, 0].plot(hist, color=color, alpha=0.7)
        axes[1, 0].set_title('Color Histograms')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend(['Blue', 'Green', 'Red'])
        
        # Feature summary
        features = analysis['features']
        feature_text = (
            f"Green Ratio: {features.get('green_ratio', 0):.2%}\n"
            f"Brown/Yellow: {features.get('brown_yellow_ratio', 0):.2%}\n"
            f"Brown Spots: {features.get('brown_spot_ratio', 0):.2%}\n"
            f"Edge Density: {features.get('edge_density', 0):.2%}\n"
            f"Texture Contrast: {features.get('contrast', 0):.1f}\n"
            f"Disease Spots: {features.get('num_disease_spots', 0)}"
        )
        axes[1, 1].text(0.1, 0.5, feature_text, fontsize=12, family='monospace',
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Extracted Features')
        axes[1, 1].axis('off')
        
        # Classification result
        result = analysis['result']
        color = 'green' if result['is_healthy'] else 'red'
        result_text = (
            f"Classification: {result['classification']}\n"
            f"Health Score: {result['health_score']:.1f}/100\n\n"
            f"Analysis:\n" + "\n".join([f"• {r}" for r in result['reasons']])
        )
        axes[1, 2].text(0.1, 0.5, result_text, fontsize=11, family='monospace',
                       verticalalignment='center', transform=axes[1, 2].transAxes,
                       color=color, weight='bold')
        axes[1, 2].set_title('Classification Result')
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Plant Health Analysis: {os.path.basename(analysis['image_path'])}", 
                    fontsize=14, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        return fig


def evaluate_dataset(detector, healthy_dir, unhealthy_dir, sample_size=None, show_confusion_matrix=True, title="Evaluation"):
    """
    Evaluate the detector on a dataset and calculate accuracy.
    Returns results including confusion matrix data.
    """
    results = {
        'healthy': {'correct': 0, 'total': 0, 'predictions': []},
        'unhealthy': {'correct': 0, 'total': 0, 'predictions': []}
    }
    
    # For confusion matrix: [actual][predicted]
    # TP = healthy predicted as healthy
    # TN = unhealthy predicted as unhealthy  
    # FP = unhealthy predicted as healthy
    # FN = healthy predicted as unhealthy
    confusion = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    # Process healthy images
    healthy_files = [f for f in os.listdir(healthy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if sample_size:
        healthy_files = healthy_files[:sample_size]
    
    print(f"\nProcessing {len(healthy_files)} healthy images...")
    for i, filename in enumerate(healthy_files):
        try:
            filepath = os.path.join(healthy_dir, filename)
            analysis = detector.analyze_image(filepath)
            predicted_healthy = analysis['result']['is_healthy']
            
            if predicted_healthy:
                confusion['TP'] += 1  # True Positive (healthy correctly identified)
            else:
                confusion['FN'] += 1  # False Negative (healthy misclassified as unhealthy)
            
            results['healthy']['correct'] += int(predicted_healthy)
            results['healthy']['total'] += 1
            results['healthy']['predictions'].append({
                'file': filename,
                'correct': predicted_healthy,
                'score': analysis['result']['health_score']
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(healthy_files)}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    # Process unhealthy images
    unhealthy_files = [f for f in os.listdir(unhealthy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if sample_size:
        unhealthy_files = unhealthy_files[:sample_size]
    
    print(f"\nProcessing {len(unhealthy_files)} unhealthy images...")
    for i, filename in enumerate(unhealthy_files):
        try:
            filepath = os.path.join(unhealthy_dir, filename)
            analysis = detector.analyze_image(filepath)
            predicted_healthy = analysis['result']['is_healthy']
            
            if not predicted_healthy:
                confusion['TN'] += 1  # True Negative (unhealthy correctly identified)
            else:
                confusion['FP'] += 1  # False Positive (unhealthy misclassified as healthy)
            
            results['unhealthy']['correct'] += int(not predicted_healthy)
            results['unhealthy']['total'] += 1
            results['unhealthy']['predictions'].append({
                'file': filename,
                'correct': not predicted_healthy,
                'score': analysis['result']['health_score']
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(unhealthy_files)}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    # Store confusion matrix
    results['confusion'] = confusion
    
    # Calculate metrics
    total_correct = results['healthy']['correct'] + results['unhealthy']['correct']
    total_samples = results['healthy']['total'] + results['unhealthy']['total']
    
    healthy_accuracy = results['healthy']['correct'] / results['healthy']['total'] * 100 if results['healthy']['total'] > 0 else 0
    unhealthy_accuracy = results['unhealthy']['correct'] / results['unhealthy']['total'] * 100 if results['unhealthy']['total'] > 0 else 0
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    
    # Calculate additional metrics
    precision = confusion['TP'] / (confusion['TP'] + confusion['FP']) if (confusion['TP'] + confusion['FP']) > 0 else 0
    recall = confusion['TP'] / (confusion['TP'] + confusion['FN']) if (confusion['TP'] + confusion['FN']) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nHealthy Detection Accuracy:   {healthy_accuracy:.1f}% ({results['healthy']['correct']}/{results['healthy']['total']})")
    print(f"Unhealthy Detection Accuracy: {unhealthy_accuracy:.1f}% ({results['unhealthy']['correct']}/{results['unhealthy']['total']})")
    print(f"\nOverall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print("="*60)
    
    # Show confusion matrix
    if show_confusion_matrix:
        plot_confusion_matrix(confusion, title=title, 
                             total_healthy=results['healthy']['total'],
                             total_unhealthy=results['unhealthy']['total'])
    
    return results


def plot_confusion_matrix(confusion, title="Confusion Matrix", total_healthy=0, total_unhealthy=0):
    """
    Plot confusion matrix visualization.
    """
    # Create confusion matrix array
    # Rows: Actual (Healthy, Unhealthy)
    # Cols: Predicted (Healthy, Unhealthy)
    cm = np.array([
        [confusion['TP'], confusion['FN']],  # Actual Healthy
        [confusion['FP'], confusion['TN']]   # Actual Unhealthy
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Healthy', 'Unhealthy']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual Label',
           xlabel='Predicted Label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Calculate percentage
            if i == 0:  # Actual Healthy
                pct = cm[i, j] / total_healthy * 100 if total_healthy > 0 else 0
            else:  # Actual Unhealthy
                pct = cm[i, j] / total_unhealthy * 100 if total_unhealthy > 0 else 0
            
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    # Add labels for TP, TN, FP, FN
    labels = [['TP\n(True Positive)', 'FN\n(False Negative)'],
              ['FP\n(False Positive)', 'TN\n(True Negative)']]
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.35, labels[i][j],
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "gray",
                   fontsize=8)
    
    # Calculate and display metrics
    total = cm.sum()
    accuracy = (confusion['TP'] + confusion['TN']) / total * 100 if total > 0 else 0
    
    metrics_text = f"Accuracy: {accuracy:.1f}%"
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    return fig


def main():
    """
    Main function to demonstrate the plant health detector.
    """
    print("="*60)
    print("PLANT HEALTH DETECTION SYSTEM")
    print("Digital Image Processing - No AI/ML")
    print("="*60)
    
    # Initialize detector
    detector = PlantHealthDetector()
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    healthy_dir = os.path.join(base_dir, "healthy")
    unhealthy_dir = os.path.join(base_dir, "unhealthy")
    
    # Test set paths
    healthy_test_dir = os.path.join(base_dir, "healthy_test")
    unhealthy_test_dir = os.path.join(base_dir, "unhealthy_test")
    
    # Interactive menu
    while True:
        print("\n" + "-"*40)
        print("OPTIONS:")
        print("1. Analyze a single image")
        print("2. Evaluate on sample (30+30) - TRAINING")
        print("3. Evaluate on full (78+78) - TRAINING")
        print("4. Evaluate on TEST SET (30+30) - UNSEEN")
        print("5. Compare healthy vs unhealthy sample")
        print("6. Exit")
        print("-"*40)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\nAvailable directories:")
            print(f"  Healthy: {healthy_dir}")
            print(f"  Unhealthy: {unhealthy_dir}")
            image_path = input("\nEnter image path (or filename from above dirs): ").strip()
            
            # Check if it's just a filename
            if not os.path.isabs(image_path):
                if os.path.exists(os.path.join(healthy_dir, image_path)):
                    image_path = os.path.join(healthy_dir, image_path)
                elif os.path.exists(os.path.join(unhealthy_dir, image_path)):
                    image_path = os.path.join(unhealthy_dir, image_path)
            
            if os.path.exists(image_path):
                print(f"\nAnalyzing: {image_path}")
                analysis = detector.analyze_image(image_path)
                detector.visualize_analysis(analysis)
            else:
                print(f"Error: File not found: {image_path}")
        
        elif choice == "2":
            print("\nEvaluating on sample (30 images per class) - TRAINING SET...")
            evaluate_dataset(detector, healthy_dir, unhealthy_dir, sample_size=30, 
                           title="Training Sample (30+30)")
        
        elif choice == "3":
            print("\nEvaluating on full TRAINING dataset...")
            evaluate_dataset(detector, healthy_dir, unhealthy_dir, sample_size=None,
                           title="Full Training Set (78+78)")
        
        elif choice == "4":
            print("\n" + "="*60)
            print("TEST SET EVALUATION (Unseen Data)")
            print("="*60)
            if os.path.exists(healthy_test_dir) and os.path.exists(unhealthy_test_dir):
                evaluate_dataset(detector, healthy_test_dir, unhealthy_test_dir, sample_size=None,
                               title="Test Set - Unseen Data (30+30)")
            else:
                print("Error: Test directories not found!")
                print(f"  Expected: {healthy_test_dir}")
                print(f"  Expected: {unhealthy_test_dir}")
        
        elif choice == "5":
            # Get first healthy and unhealthy sample
            healthy_files = os.listdir(healthy_dir)
            unhealthy_files = os.listdir(unhealthy_dir)
            
            if healthy_files and unhealthy_files:
                healthy_path = os.path.join(healthy_dir, healthy_files[0])
                unhealthy_path = os.path.join(unhealthy_dir, unhealthy_files[0])
                
                print("\n--- Healthy Sample Analysis ---")
                healthy_analysis = detector.analyze_image(healthy_path)
                print(f"File: {healthy_files[0]}")
                print(f"Result: {healthy_analysis['result']['classification']}")
                print(f"Score: {healthy_analysis['result']['health_score']:.1f}/100")
                
                print("\n--- Unhealthy Sample Analysis ---")
                unhealthy_analysis = detector.analyze_image(unhealthy_path)
                print(f"File: {unhealthy_files[0]}")
                print(f"Result: {unhealthy_analysis['result']['classification']}")
                print(f"Score: {unhealthy_analysis['result']['health_score']:.1f}/100")
                
                # Visualize both
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                healthy_rgb = cv2.cvtColor(healthy_analysis['processed_images']['original'], cv2.COLOR_BGR2RGB)
                axes[0].imshow(healthy_rgb)
                axes[0].set_title(f"HEALTHY\nScore: {healthy_analysis['result']['health_score']:.1f}", color='green', fontsize=14)
                axes[0].axis('off')
                
                unhealthy_rgb = cv2.cvtColor(unhealthy_analysis['processed_images']['original'], cv2.COLOR_BGR2RGB)
                axes[1].imshow(unhealthy_rgb)
                axes[1].set_title(f"UNHEALTHY\nScore: {unhealthy_analysis['result']['health_score']:.1f}", color='red', fontsize=14)
                axes[1].axis('off')
                
                plt.suptitle("Healthy vs Unhealthy Comparison", fontsize=16, weight='bold')
                plt.tight_layout()
                plt.show()
        
        elif choice == "6":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

