import matplotlib.pyplot as plt

import numpy as np

def plot_rr_distribution(rr_intervals):
    print("\nRR Interval Stats (seconds)")
    print("Total beats:", len(rr_intervals))
    print("Mean:", np.mean(rr_intervals))
    print("Median:", np.median(rr_intervals))
    print("Min:", np.min(rr_intervals))
    print("Max:", np.max(rr_intervals))

    plt.figure(figsize=(10, 5))
    plt.hist(rr_intervals, bins=100)
    plt.xlabel("RR Interval (seconds)")
    plt.ylabel("Count")
    plt.title("RR Interval Distribution (All Records)")
    plt.grid(True)
    plt.show()
