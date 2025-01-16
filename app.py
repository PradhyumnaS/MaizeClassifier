import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaizeKernelAnalyzer:
    def __init__(self):
        self.model = self.load_model()
        
        self.reference_hsv = np.array([30, 220, 200])
        self.quality_criteria = {
            "excellent": {
                "h": (0, 65),
                "s": (200, 255),
                "v": (180, 255)
            },
            "good": {
                "h": (65, 90),
                "s": (150, 255),
                "v": (150, 255)
            },
            "poor": {
                "h": (90, 355),
                "s": (0, 255),
                "v": (0, 255)
            }
        }
        
        self.color_normalizer = ColorNormalizer()

    def load_model(self):
        try:
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect_kernels(self, image):
        try:
            results = self.model(image, conf=0.02)[0]
            return results.boxes
        except Exception as e:
            logger.error(f"Error in kernel detection: {e}")
            return None

    def analyze_kernel(self, kernel_region):
        try:
            kernel_hsv = cv2.cvtColor(kernel_region, cv2.COLOR_BGR2HSV)
            normalized_hsv = self.color_normalizer.normalize_hsv(kernel_hsv)
            
            hsv_mean = cv2.mean(normalized_hsv)[:3]
            
            quality = self.determine_quality(hsv_mean)
            
            metrics = {
                "hsv_values": [float(x) for x in hsv_mean],
                "quality": quality,
                "confidence": self.calculate_quality_confidence(hsv_mean, quality)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error in kernel analysis: {e}")
            return None

    def determine_quality(self, hsv_values):
        for quality, criteria in self.quality_criteria.items():
            if (criteria["h"][0] <= hsv_values[0] <= criteria["h"][1] and
                criteria["s"][0] <= hsv_values[1] <= criteria["s"][1] and
                criteria["v"][0] <= hsv_values[2] <= criteria["v"][1]):
                return quality
        return "poor"

    def calculate_quality_confidence(self, hsv_values, quality):
        criteria = self.quality_criteria[quality]
        h_range = criteria["h"][1] - criteria["h"][0]
        s_range = criteria["s"][1] - criteria["s"][0]
        v_range = criteria["v"][1] - criteria["v"][0]
        
        h_conf = 1 - (abs(hsv_values[0] - np.mean(criteria["h"])) / (h_range/2))
        s_conf = 1 - (abs(hsv_values[1] - np.mean(criteria["s"])) / (s_range/2))
        v_conf = 1 - (abs(hsv_values[2] - np.mean(criteria["v"])) / (v_range/2))
        
        return float(np.clip(np.mean([h_conf, s_conf, v_conf]), 0, 1))

class ColorNormalizer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.scaler = StandardScaler()

    def normalize_image(self, image):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            l_norm = self.clahe.apply(l)
            
            a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
            b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
            
            normalized_lab = cv2.merge([l_norm, a_norm, b_norm])
            normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
            
            return normalized_bgr
        except Exception as e:
            logger.error(f"Error in image normalization: {e}")
            return image

    def normalize_hsv(self, hsv_image):
        try:
            h, s, v = cv2.split(hsv_image)
            
            h_norm = cv2.normalize(h, None, 0, 179, cv2.NORM_MINMAX)
            s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
            v_norm = self.clahe.apply(v)
            
            return cv2.merge([h_norm, s_norm, v_norm])
        except Exception as e:
            logger.error(f"Error in HSV normalization: {e}")
            return hsv_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('static/uploads')
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

analyzer = MaizeKernelAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return json.dumps({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return json.dumps({'error': 'No image selected'}), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        image = cv2.imread(str(filepath))
        normalized_image = analyzer.color_normalizer.normalize_image(image)
        
        boxes = analyzer.detect_kernels(normalized_image)
        results = []
        
        output_image = normalized_image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            kernel_region = normalized_image[y1:y2, x1:x2]
            
            analysis = analyzer.analyze_kernel(kernel_region)
            if analysis:
                results.append({
                    "kernel_id": i + 1,
                    "position": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "analysis": analysis
                })
                
                color = {
                    "excellent": (0, 255, 0),
                    "good": (0, 255, 255),
                    "poor": (0, 0, 255)
                }.get(analysis["quality"], (0, 0, 255))
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(output_image, 
                          f"#{i+1}",
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        output_filename = f"analyzed_{filename}"
        output_path = app.config['UPLOAD_FOLDER'] / output_filename
        cv2.imwrite(str(output_path), output_image)
        
        return json.dumps({
            'original_image': f'uploads/{filename}',
            'analyzed_image': f'uploads/{output_filename}',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        return json.dumps({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)