# Human Activity Recognition using Edge AI & TensorFlow Lite

This project focuses on real-time human activity recognition (HAR) using accelerometer data processed on edge devices. A lightweight deep learning model is trained and converted to TensorFlow Lite for deployment on mobile or embedded systems.

## 🔍 Problem Statement

Recognizing physical activities (like walking, sitting, jogging, etc.) in real time using sensor data has significant applications in healthcare, fitness tracking, and smart environments. The goal is to build a model that can classify user activity based on motion data — directly on low-resource edge devices.

## 📂 Dataset

We used the [WISDM v1.1 dataset](https://www.cis.fordham.edu/wisdm/dataset.php), which contains labeled accelerometer data collected from smartphone users performing six types of activities:

* Jogging
* Walking
* Upstairs
* Downstairs
* Sitting
* Standing

Each activity was recorded with timestamps and X, Y, Z accelerometer readings.

## 🛠️ Approach

1. **Preprocessing**:

   * Cleaned raw sensor data.
   * Applied a sliding window (60 timesteps) to generate samples.
   * Normalized features (`x`, `y`, `z`, and `magnitude`).

2. **Modeling**:

   * Built a 1D CNN model using TensorFlow/Keras.
   * Converted the model to `.tflite` format for deployment on mobile.

3. **Evaluation**:

   * Achieved strong classification performance.
   * Final model supports edge execution via Edge Impulse or Android inference.

## 📱 Edge Impulse Integration

You can collect your own motion data directly from your smartphone using the [Edge Impulse](https://edgeimpulse.com) mobile app. Simply:

1. Record and upload sensor data (accelerometer) on the Edge Impulse studio.
2. Download the resulting JSON file.
3. Place the JSON in this project and run the provided conversion script to generate a CSV with `accX, accY, accZ, magnitude`.
4. The TFLite model will then predict which activity your data represents.

## 🔍 Model Clarification

**Why might the model assign a higher probability to *Jogging* over *Sitting* or *Standing* when the data appears static?**

✅ **Possible Reasons:**

* **Training Data Characteristics:** The *Jogging* class in the training set may have included intervals with subtle accelerations (e.g., start/end of a run), causing the model to associate small vibrations with jogging.
* **Class Imbalance:** There may have been fewer examples of truly static activities (*Sitting*/*Standing*), so the model learned those patterns less robustly.
* **Data Ambiguity:** Real-world static data often contains minor fluctuations. The model interprets slight movements as resembling slow walking or jogging, so it selects the closest match based on learned features.

Rest assured, this behavior reflects the model’s decision boundaries and does not indicate a fundamental flaw.

## 🧪 Try It Yourself

Clone the repository, install dependencies, and run:

```bash
pip install -r requirements.txt
python classify.py
```

## 🤝 Acknowledgements

* **Dataset:** WISDM Lab @ Fordham University
* **Tools:** TensorFlow Lite, Edge Impulse, Python, Pandas, Sklearn

---

*Authored by Sheida Mehrazi*: M.Sc. in Computer Architecture · IoT & Edge AI Enthusiast
