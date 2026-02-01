import wfdb
from scipy.signal import resample_poly # uses polyphase filtering → much better for ECG morphology

# ----------------------------------------------- Function to load and resample record -----------------------------------------------

def load_record(record_path, lead=0):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, lead]
    fs = record.fs
    samples = ann.sample        # array of integers representing the sample indices (positions) in the original signal where each annotation occurs.
    aux_notes = ann.aux_note    # used for rhythm state change

    # Resample to fixed 360 Hz if different
    target_fs = 360
    if fs != target_fs:
        signal = resample_poly(signal, target_fs, fs)
        samples = (samples * target_fs / fs).astype(int)
        fs = target_fs

    return signal, samples, aux_notes, fs



