import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Rate limiting to prevent abuse
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["100 per day", "10 per minute"])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration (prefer GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model definition
class CombinedModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CombinedModel, self).__init__()
        self.efficientnet = nn.Sequential(*list(models.efficientnet_b3(weights=None).children())[:-1])
        self.vit = models.vit_b_16(weights=None)
        self.classifier = nn.Linear(2536, num_classes)  # Adjust if dimensions differ

    def forward(self, x):
        eff_features = self.efficientnet(x)
        eff_features = eff_features.view(eff_features.size(0), -1)
        vit_features = self.vit(x)
        combined = torch.cat((eff_features, vit_features), dim=1)
        output = self.classifier(combined)
        return output

# Load model
try:
    logger.info("Loading model...")
    model = CombinedModel().to(device)
    state_dict = torch.load('best_combined_model.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")  # Limit prediction requests
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved file: {filename}")

        # Process image
        img = Image.open(filepath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        result = "Fake" if predicted.item() == 1 else "Real"
        confidence = probabilities[0][predicted.item()].item() * 100

        # Clean up uploaded file (optional, depending on storage needs)
        # os.remove(filepath)

        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'image_url': f"/static/uploads/{filename}"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
