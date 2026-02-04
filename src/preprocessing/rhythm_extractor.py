import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) # include the src directory in the path

from loaders.rhythm_rec_finder import get_all_records
from loaders.rhythm_wfdb_loader import load_record

target_fs = 360
window_sec = 5
af_threshold = 0.5  # % of window that must be AF to label as AF here we set it to 50 %

# ----------------------------------------------- Extract AF rhythm intervals -----------------------------------------------

def extract_af_intervals(samples, aux_notes, signal_len):
    af_intervals = []
    af_start = None

    for i, note in enumerate(aux_notes):
        if note is None:
            continue

        note = note.strip().upper()  # Case-insensitive check

        # Start of AF
        if "AF" in note:
            af_start = samples[i]

        # End of AF, back to Normal/Sinus
        if ("N" in note or "SR" in note) and af_start is not None:
            af_intervals.append((af_start, samples[i]))
            af_start = None

    # If AF continues until end of signal
    if af_start is not None:
        af_intervals.append((af_start, signal_len))

    return af_intervals


# ----------------------------------------------- Overlap calculation -----------------------------------------------

def compute_overlap(window_start, window_end, af_intervals):
    overlap = 0

    for af_start, af_end in af_intervals:
        start = max(window_start, af_start)
        end = min(window_end, af_end)
        if start < end:
            overlap += (end - start)

    return overlap

# ----------------------------------------------- Extract rhythm windows -----------------------------------------------

def extract_rhythm_windows(records):
    X = []
    y = []

    for rec_path in records:
        try:
            signal, samples, aux_notes, fs = load_record(rec_path)

            af_intervals = extract_af_intervals(
                samples, aux_notes, len(signal)
            )

            window_size = int(window_sec * fs)
            step = window_size

            for start in range(0, len(signal) - window_size, step):
                end = start + window_size
                segment = signal[start:end]

                overlap = compute_overlap(start, end, af_intervals)
                af_ratio = overlap / window_size

                label = 1 if af_ratio >= af_threshold else 0

                X.append(segment)
                y.append(label)

        except Exception as e:
            print("Skipped:", rec_path, e)

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    return X, y

    # ----------------------------------------------- Main Script -----------------------------------------------

if __name__ == "__main__":
    base_path = "data/raw/afdb"

    records = get_all_records(base_path)
    print("AFDB Records found:", len(records))

    X, y = extract_rhythm_windows(records)

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/rhythm_X.npy", X)
    np.save("data/processed/rhythm_y.npy", y)

    print("Saved AFDB processed data")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class distribution:", np.unique(y, return_counts=True))
    # Quick AF window check
    af_windows = np.sum(y==1)
    print("Number of AF windows:", af_windows)




