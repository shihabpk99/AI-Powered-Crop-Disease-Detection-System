import tensorflow as tf


print("--- SYSTEM CHECK ---")
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

if gpus:
    print(f"SUCCESS: GPU detected: {gpus}")
    print("Training will use the available GPU.")
else:
    print("WARNING: No GPU detected. TensorFlow is running on CPU.")
    print("Training will be significantly slower.")
    print(f"Devices found: {cpus}")

try:
    import cv2  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401
    import numpy as np  # noqa: F401
    from sklearn.metrics import classification_report  # noqa: F401

    print("SUCCESS: Matplotlib, NumPy, OpenCV, and scikit-learn are installed.")
except ImportError as error:
    print(f"ERROR: Missing library -> {error.name}")
    print(f"Install it with: pip install {error.name}")

print("--------------------")
