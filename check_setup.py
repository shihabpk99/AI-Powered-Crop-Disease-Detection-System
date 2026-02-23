import tensorflow as tf
import os

print("--- SYSTEM CHECK ---")

# 1. Check TensorFlow Version
print(f"TensorFlow Version: {tf.__version__}")

# 2. Check for GPU
# This lists the hardware TF can see.
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if gpus:
    print(f"✅ SUCCESS: GPU Detected: {gpus}")
    print("Training will be fast.")
else:
    print("⚠️ WARNING: No GPU detected. TensorFlow is running on CPU.")
    print("Training will be significantly slower.")
    print(f"Devices found: {cpus}")

# 3. Check for Helper Libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from sklearn.metrics import classification_report
    print("✅ SUCCESS: All helper libraries (Matplotlib, Numpy, OpenCV, Sklearn) are installed.")
except ImportError as e:
    print(f"❌ ERROR: Missing library -> {e.name}")
    print(f"Please run: pip install {e.name}")

print("--------------------")