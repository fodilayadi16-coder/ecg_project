import wfdb
from scipy.signal import resample_poly

def load_record(record_path, lead=0, target_fs=360):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, lead]
    r_peaks = ann.sample
    symbols = ann.symbol
    fs = record.fs

    # Resample to target_fs if needed
    target_fs = 360
    if fs != target_fs:
        signal = resample_poly(signal, target_fs, fs)
        r_peaks = (r_peaks * target_fs / fs).astype(int)
        fs = target_fs

    return signal, r_peaks, symbols, fs
