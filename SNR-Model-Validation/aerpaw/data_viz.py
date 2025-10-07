import numpy as np
import matplotlib.pyplot as plt

def main():
    snrs = []
    with open("2025-10-06_15_19_12_snr_log.txt", "r") as f:
        for line in f.readlines():
            tokens = line.split( )
            if len(tokens) == 4 and float(tokens[-1]) > 0: 
                snrs.append(float(tokens[-1]))
    
    snrs = np.array(snrs).astype(np.float64)
    print(len(snrs))
    print(snrs[:10])
    plt.plot(snrs, color="red")
    plt.show()

if __name__ == '__main__':
    main()
