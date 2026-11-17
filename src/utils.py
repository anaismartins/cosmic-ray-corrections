import numpy as np
import globals as g


def chi2(res):
    chi2 = np.sum((res) ** 2)
    return chi2

def normalized_chi2(res):
    sigma0 = np.std(res)
    chi2 = np.sum((res / sigma0) ** 2)
    N = len(res)
    norm_chi2 = (chi2 - N) / np.sqrt(2 * N)
    return norm_chi2

def log_params(log_file):
    with open(log_file, 'a') as f:
        f.write("Parameters:\n")
        f.write(f"FIGURES_PATH: {g.FIGURES_PATH}\n")
        f.write(f"DATA_PATH: {g.DATA_PATH}\n")
        f.write(f"LOG_PATH: {g.LOG_PATH}\n\n")
        f.write(f"BAND_LIST: {g.BAND_LIST}\n")
        f.write(f"GLITCH_TYPES: {g.GLITCH_TYPES}\n\n")
        f.write(f"SAMPRATE: {g.SAMPRATE}\n")
        f.write(f"SIGMA: {g.SIGMA}\n\n")
        f.write(f"FAST_PART: {g.FAST_PART}\n")
        f.write(f"NO_TRUST: {g.NO_TRUST}\n\n")
        f.write(f"CUT_OFF: {g.CUT_OFF}\n\n")
        f.write(f"NSECS: {g.NSECS}\n")
        f.write(f"CLASSIFY_SECS_FIRST: {g.CLASSIFY_SECS_FIRST}\n")
        f.write(f"CLASSIFY_SECS: {g.CLASSIFY_SECS}\n\n")
        f.write(f"MAXITER: {g.MAXITER}\n\n")
        f.write(f"PLOTS: {g.PLOTS}\n\n")
        f.write("--------------------------------------------------\n\n")

def log_output(log_file, message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')