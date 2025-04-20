import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess import prepare_input

# لود داده ضبط‌شده از Edge Impulse
df = pd.read_csv("data/sample1.csv")

# فرض بر اینه که فایل دارای ستون‌های x, y, z هست
x = df['accelerometerX'].values
y = df['accelerometerY'].values
z = df['accelerometerZ'].values

# محاسبه magnitude
magnitude = np.sqrt(x**2 + y**2 + z**2)

# ساخت لیست flat به صورت [x0, y0, z0, m0, ...]
features = np.array([x, y, z, magnitude]).T.flatten()

# آماده‌سازی داده برای مدل
input_tensor = prepare_input(features, expected_timesteps=100, num_features=4)

# اجرای مدل TFLite
interpreter = tf.lite.Interpreter(model_path="activity_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print("Predicted class:", np.argmax(output))
