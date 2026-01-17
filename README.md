## Face Emotions with YOLOv11

This project is a computer vision–based emotion detection system built using the YOLOv11 model.
The model is trained to classify six basic human emotions based on Paul Ekman’s theory of emotions, which are:

😠 Angry

🤢 Disgust

😨 Fear

😊 Happy

😢 Sad

😲 Surprise

The system detects faces in real-time or from images and classifies the detected facial expression into one of these emotion categories. Through model optimization and dataset refinement, the system achieves up to 98% emotion recognition accuracy on the validation dataset, demonstrating strong performance for real-time applications and research purposes.

### 🧠 Model Performance

The YOLOv11 model was trained and evaluated on a labeled facial expression dataset.
After dataset cleaning and model tuning, the system achieved:

1. Accuracy: up to 98% on the validation dataset
2. Fast inference speed suitable for real-time detection
3. Stable emotion classification across different lighting conditions

Note: Accuracy may vary depending on input quality and environment.

### ⚙️ Technology Stack
| Category                | Technology / Library   | Description                                                     |
| ----------------------- | ---------------------- | --------------------------------------------------------------- |
| Programming Language    | Python 3.x             | Main language used for model training and inference             |
| Deep Learning Framework | PyTorch                | Framework for building and training deep learning models        |
| Object Detection Model  | YOLOv11 (Ultralytics)  | Real-time object detection and emotion classification model     |
| Computer Vision         | OpenCV                 | Image and video processing for face detection and visualization |
| Numerical Computing     | NumPy                  | Efficient numerical operations for image and tensor processing  |
| Model Training          | Ultralytics YOLO       | Training, validation, and inference pipeline                    |
| Dataset Handling        | Custom Emotion Dataset | Labeled facial expression dataset based on Paul Ekman’s theory  |
| Model Weights           | last.pt                | Trained YOLOv11 weights for emotion detection                   |
| Visualization           | Matplotlib             | Visualization of training results and model performance         |
| Environment             | Windows / Linux        | Development and training environment                            |
| Version Control         | Git & GitHub           | Source code management and collaboration                        |
| Dependency Management   | pip & requirements.txt | Managing Python dependencies                                    |
| Deployment (Optional)   | Webcam / Image Input   | Real-time and static image emotion detection                    |

## 📂 Project Structure

FaceEmotions-YOLOv11/
│
├── train/                     # Training dataset (images & labels)
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   └── surprise/
│
├── runs/                      # YOLO training & validation results
│   └── detect/
│
├── yolo_emotions.py           # Main script for emotion detection
├── last.pt                    # Trained YOLOv11 model weights
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation


