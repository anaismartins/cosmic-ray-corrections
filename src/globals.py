FIGURES_PATH = "../output/figures/"
DATA_PATH = "../data/"
LOG_PATH = "../output/logs/"

BAND_LIST = ["100-1a"]#, "100-1b", "100-2a", "100-2b", "100-3a", "100-3b", "100-4a", "100-4b"]
GLITCH_TYPES = ["short"]#, "long", "slow"]

SAMPRATE = 180.3737 #Hz
SIGMA = 0.01 #random std for the white noise

FAST_PART = 0 # seconds
NO_TRUST = 3 # samples

CUT_OFF = 10 # for finding the peaks

NSECS = 1 # used for matched filter, estimating template etc
CLASSIFY_SECS_FIRST = 0.1 # seconds, used for first iteration of classification
CLASSIFY_SECS = 0.25

MAXITER = 100

PLOTS = True