import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.model_selection import train_test_split
from models.rhythm_cnn import build_rhythm_cnn
from training.callbacks import make_callbacks
# from utils.balancing import get_class_weights
from utils.oversampling import moderate_ros

X = np.load("data/processed/rhythm_X.npy")
y = np.load("data/processed/rhythm_y.npy")

print("Original classes:", np.unique(y, return_counts=True))

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2) # stratify=y keeps the same class proportions in both sets

# Apply moderate ROS only on training set
X_train_resampled, y_train_resampled = moderate_ros(X_train, y_train)


model = build_rhythm_cnn()
model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
  # class_weight = get_class_weights(y_train), # A dictionary where keys are class labels and values are number of samples in that class
    callbacks=make_callbacks("rhythm")
)

model.save("models/rhythm_final.keras")