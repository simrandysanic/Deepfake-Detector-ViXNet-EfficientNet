# Deepfake Detection using ViXNet + EfficientNet-B3

CS-512 Major Project
Indian Institute of Technology Ropar

> An efficient and practical deepfake image detection system built by adapting the ViXNet architecture — replacing Xception with EfficientNet-B3 to optimize performance for real-world applications.

---

## Abstract

This project addresses the growing challenge of detecting image-based deepfakes. We adapt the ViXNet architecture by replacing the heavyweight Xception backbone with **EfficientNet-B3**, significantly improving computational efficiency while retaining strong detection performance.

The dual-branch model combines:
- A **Vision Transformer (ViT-B16)** for patch-based local/global relationships
- A **CNN path (EfficientNet-B3)** for spatial features

> **Achieved Accuracy**: ~63% on FaceForensics++ (FF++)  
> **Key Advantage**: 2–3× faster inference and ~50% fewer parameters than ViXNet (Xception)

---

## Report

You can find the full technical report here:  
[`Deepfake_Detection_Project_Report.pdf`]([./report/Deepfake_Detection_Project_Report.pdf](https://github.com/simrandysanic/Deepfake-Detector-ViXNet-EfficientNet/blob/main/Report.pdf))

It includes:
- Model architecture
- Dataset details (FF++, Celeb-DF)
- Implementation specifics
- Results and discussion
- Limitations and future work

---

## Datasets

- **FaceForensics++ (FF++)** – Real/fake face videos using various synthesis methods  
- **Celeb-DF** – High-quality deepfake videos of public figures

(*Note: Follow official sources for access.*)

---

## Key Features

- Dual-branch model: Combines ViT and EfficientNet for robust detection
- EfficientNet-B3 backbone: Lightweight yet effective CNN
- Web interface to upload and analyze face images 
- Modular design for future enhancements
  
---

## Contributors

- Divya Chauhan – 2024CSM1006  
- Ghulam Haider – 2024CSM1008  
- **Simran Prasad** – 2024CSM1018  
- Yogeshwar – 2024CSM1021  

