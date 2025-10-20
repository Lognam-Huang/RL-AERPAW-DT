import numpy as np
import matplotlib.pyplot as plt

def main():
    snrs = []
    with open("experiment_data/2025-10-19_16_13_57_snr_log.txt", "r") as f:
        for line in f.readlines():
            tokens = line.split( )
            if len(tokens) == 4 and float(tokens[-1]) > 0: 
                snrs.append(float(tokens[-1]))
    
    snrs = np.array(snrs).astype(np.float64)
    print(len(snrs) / (256))  # Sample rate per second
    print(snrs[:10])

    # Taking the moving average
    window = 1000  # Window size
    weight = np.ones(window) / window
    snrs = np.convolve(snrs, weight, mode="valid")
    plt.plot(snrs, color="red")
    plt.show()

if __name__ == '__main__':
    main()
