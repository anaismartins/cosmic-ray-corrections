"""
In this script, I want to try out the way that the Planck team used to remove the glitches.
"""

import globals as g
import numpy as np
import templates as templates
from scipy.signal import find_peaks
import utils

def matched_filter(res, glitch_params=None):
    res = np.asarray(res, dtype=float)
    glitch_model = templates.glitch_model_func(np.arange(0, g.NSECS, 1 / g.SAMPRATE), 1,
                                               glitch_params=glitch_params)
    # make the matched filter zero mean and normalized
    template_norm = np.linalg.norm(glitch_model - np.mean(glitch_model))
    if len(res) < len(glitch_model):
        raise ValueError("The TOD must be longer than the glitch template")

    # Correlate each possible onset with the causal template.  Reversing the
    # template converts correlation into convolution; the valid slice has
    # one score for each possible template start.
    fft_length = len(res) + len(glitch_model) - 1
    correlation = np.fft.irfft(np.fft.rfft(res, fft_length) * np.fft.rfft(glitch_model[::-1],
                                                                          fft_length), fft_length)
    score = correlation[len(glitch_model) - 1:len(res)] / (g.SIGMA * template_norm)

    glitch_idx, _ = find_peaks(score, height=g.CUT_OFF, prominence=5)

    return glitch_idx, score

if __name__ == "__main__":
    res = np.load(f"{g.DATA_PATH}143-2a_simulations.npy")
    matched_filter(res)
    