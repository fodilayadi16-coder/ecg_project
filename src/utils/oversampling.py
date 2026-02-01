from imblearn.over_sampling import RandomOverSampler
import numpy as np

def moderate_ros(X, y, target_ratio=0.5, random_state=42):
    """
    Moderately oversample minority classes.
    target_ratio: float, proportion of majority class to oversample minority classes to
    """
    unique, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    
    # Build sampling_strategy dict
    sampling_strategy = {}
    for cls, count in zip(unique, counts):
        if count < majority_count:
            # increase minority to target_ratio * majority_count
            sampling_strategy[cls] = int(count + (majority_count - count) * target_ratio)
    
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(len(X), -1), y)  # reshape if X is (N,180,1)
    X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])

    print("Before ROS:", X.shape, y.shape)
    print("After ROS:", X_resampled.shape, y_resampled.shape)
    
    return X_resampled, y_resampled


