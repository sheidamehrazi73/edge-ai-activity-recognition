
# Human Activity Recognition using Edge AI & TensorFlow Lite

This project focuses on real-time human activity recognition (HAR) using accelerometer data processed on edge devices. A lightweight deep learning model is trained and converted to TensorFlow Lite for deployment on mobile or embedded systems.

## 🔍 Problem Statement
Recognizing physical activities (like walking, sitting, jogging, etc.) in real time using sensor data has significant applications in healthcare, fitness tracking, and smart environments. The goal is to build a model that can classify user activity based on motion data — directly on low-resource edge devices.

## 📂 Dataset
We used the [WISDM v1.1 dataset](https://www.cis.fordham.edu/wisdm/dataset.php), which contains labeled accelerometer data collected from smartphone users performing six types of activities:
- Jogging
- Walking
- Upstairs
- Downstairs
- Sitting
- Standing

Each activity was recorded with timestamps and X, Y, Z accelerometer readings.

## 🛠️ Approach
1. **Preprocessing**:
   - Cleaned raw sensor data.
   - Applied a sliding window (60 timesteps) to generate samples.
   - Normalized features (x, y, z, and magnitude).

2. **Modeling**:
   - Built a 1D CNN model using TensorFlow/Keras.
   - Converted the model to `.tflite` format for deployment on mobile.

3. **Evaluation**:
   - Achieved strong classification performance.
   - Final model supports edge execution via Edge Impulse or Android inference.

## 💡 Key Features
- Designed for Edge AI deployment.
- Fully documented in Jupyter Notebook.
- Easily extendable to live mobile data collection (Edge Impulse).
- Efficient and lightweight model optimized for embedded devices.

## 🧪 Try It Yourself
Clone the repository, install dependencies, and run:

```bash
pip install -r requirements.txt
python classify.py
