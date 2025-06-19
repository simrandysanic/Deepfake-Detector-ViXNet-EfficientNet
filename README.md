# Deepfake Detection using ViXNet + EfficientNet-B3  
**CS-512 Major Project — Indian Institute of Technology Ropar**

An efficient and practical deepfake image detection system built by adapting the ViXNet architecture — replacing Xception with EfficientNet-B3 to optimize performance for real-world deployment.

---

## 🚀 Abstract

This project addresses the growing challenge of detecting image-based deepfakes. We adapt the ViXNet architecture by replacing the heavyweight Xception backbone with **EfficientNet-B3**, significantly improving computational efficiency while retaining strong detection performance.

The dual-branch model combines:
- 🔬 A **Vision Transformer (ViT-B16)** for patch-based local/global relationships
- 🧠 A **CNN path (EfficientNet-B3)** for global spatial features

**Performance:**
- ✅ Accuracy: ~63% on FaceForensics++ (FF++) test set
- ⚡ 2–3× faster inference
- 📉 ~50% fewer parameters than ViXNet (Xception)

---

## 🧠 Architecture Overview

- **Patch-Based Attention Path (ViT-B16)**: Splits face images into 16×16 patches, processes them using 3×3 convolutions and Vision Transformer.
- **Global CNN Path (EfficientNet-B3)**: Extracts spatial features using MBConv and SE blocks.
- **Feature Fusion**: Concatenates ViT and CNN outputs for classification.
- **Classification Head**: Three dense layers followed by sigmoid activation.

> Total Parameters: ~98M (ViT: 86M, EfficientNet-B3: 12M)

---

## 📊 Results

### 🧪 Intra-Dataset (FF++)
- Accuracy: ~63%
- F1 Score: ~64%
- AUC: High

### 🔁 Efficiency Gains
- Parameters: 12M (vs. 22.9M in Xception)
- FLOPs: 1.8B (vs. 8.4B)
- Inference: ~2–3× faster on GPU

---

## 🧪 Datasets

| Dataset      | Description                                      | Use                        |
|--------------|--------------------------------------------------|----------------------------|
| **FF++**     | ~1,000 real / ~4,000 fake face videos            | Training + evaluation      |
| **Celeb-DF** | ~590 real / ~5,639 fake celeb deepfake videos    | Cross-dataset validation   |

> Preprocessing included face detection (MTCNN), resizing to 300×300, and class balancing.

---

## 💻 Repository Structure

```
deepfake-detection-vixnet/
├── app.py                  # Flask web application
├── templates/
│   └── index.html          # Web UI
├── best_combined_model.pth# Trained model (download separately)
├── requirements.txt        # Python dependencies
├── train.py                # Model training script
├── preprocess.py           # Preprocessing utilities
├── evaluate.py             # Evaluation logic
├── README.md
└── report/
    └── Deepfake_Detection_Project_Report.pdf
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- NVIDIA GPU (recommended)

### Step 1: Clone and Create Environment
```bash
git clone https://github.com/simrandysanic/Deepfake-Detector-ViXNet-EfficientNet.git
cd Deepfake-Detector-ViXNet-EfficientNet
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Trained Model  
📦 [Insert your Google Drive link or say “available upon request”]  
Place `best_combined_model.pth` in the project root.

---

## 🌐 Running the Web App

```bash
python app.py
```

Visit [http://localhost:4300](http://localhost:4300)  
Upload an image → See Real/Fake prediction with confidence score.

---

## 🏋️‍♂️ Training the Model

> Note: Requires large compute (e.g., NVIDIA T4 GPU or higher)

1. Download FF++ and Celeb-DF datasets
2. Update dataset paths in `train.py`
3. Run:
```bash
python train.py
```

---

## 📎 Report

Full technical documentation:  
[`Deepfake_Detection_Project_Report.pdf`](https://github.com/simrandysanic/Deepfake-Detector-ViXNet-EfficientNet/blob/main/Report.pdf)

Includes:
- Model architecture
- Dataset usage
- Implementation details
- Performance comparison
- Limitations & future work

---

## 👨‍💻 Contributors

- Divya Chauhan — 2024CSM1006  
- Ghulam Haider — 2024CSM1008  
- **Simran Prasad** — 2024CSM1018  
- Yogeshwar — 2024CSM1021  

---

## 🧠 Acknowledgments

- ViXNet authors for their original architecture and methodology  
- FaceForensics++ and Celeb-DF dataset creators  
- Libraries used: PyTorch, TensorFlow, EfficientNet-PyTorch, Flask, Facenet-PyTorch  

---

## 📜 License

Licensed under the MIT License. See `LICENSE` for details.
