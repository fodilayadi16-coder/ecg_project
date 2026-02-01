import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------- Paths -------------------------

model_path = "models/rhythm_best.keras"
x_path = "data/processed/rhythm_X.npy"
y_path = "data/processed/rhythm_y.npy"

# ------------------------- Load data -------------------------

X = np.load(x_path)
y = np.load(y_path)  # 0 = Normal, 1 = AF

print("Full dataset shape:", X.shape, y.shape)

# -------------------- Train / Val / Test Split --------------------

# First split: Train (70%) and Temp (30%)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Second split: Val (15%) and Test (15%)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Test set shape:", X_test.shape, y_test.shape)

# -------------------- Load model --------------------

model = load_model(model_path)

# -------------------- Predict --------------------

y_pred_prob = model.predict(X_test, batch_size=256)

# Binary or multi-class safe handling

if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
    y_pred = np.argmax(y_pred_prob, axis=1)
else:
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# -------------------- Segment-level accuracy --------------------

seg_accuracy = accuracy_score(y_test, y_pred)
print(f"\nSegment-level accuracy: {seg_accuracy*100:.2f}%")

# -------------------- Confusion matrix & report --------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "AF"]))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# -------------------- AF Metrics --------------------

def af_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- Time in AF ----

    af_segments_true = y_true == 1
    af_segments_pred = y_pred == 1

    correct_af_segments = np.sum(af_segments_true & af_segments_pred)
    total_af_segments = np.sum(af_segments_true)

    time_in_af = correct_af_segments / total_af_segments if total_af_segments > 0 else 0

    # ---- Episode Detection Rate (EDR) ----

    def find_episodes(labels):
        episodes = []
        start = None
        for i, label in enumerate(labels):
            if label == 1 and start is None:
                start = i
            elif label == 0 and start is not None:
                episodes.append((start, i - 1))
                start = None
        if start is not None:
            episodes.append((start, len(labels) - 1))
        return episodes

    true_episodes = find_episodes(y_true)
    pred_episodes = find_episodes(y_pred)

    detected = 0
    for t_start, t_end in true_episodes:
        for p_start, p_end in pred_episodes:
            # Overlap check
            if p_end >= t_start and p_start <= t_end:
                detected += 1
                break

    edr = detected / len(true_episodes) if len(true_episodes) > 0 else 0

    return time_in_af, edr, true_episodes

time_af, edr, true_eps = af_metrics(y_test, y_pred)

print(f"\nTime in AF: {time_af*100:.2f}%")
print(f"Episode Detection Rate (EDR): {edr*100:.2f}%")
print(f"Total AF episodes in test set: {len(true_eps)}")

print("True AF %:", 100 * np.mean(y_test == 1))
print("Pred AF %:", 100 * np.mean(y_pred == 1))

# -------------------- Plot AF Timeline --------------------

plt.figure(figsize=(15, 3))
plt.plot(y_test, label="True AF", linewidth=2)
plt.plot(y_pred, label="Predicted AF", linestyle="--")
plt.title("AF Prediction Timeline (Test Set)")
plt.xlabel("Segment Index")
plt.ylabel("Label (0 = Normal, 1 = AF)")
plt.legend()
plt.tight_layout()
plt.savefig("af_timeline.png")
plt.show()

