WINDOWSIZE = 1000
FIGURES_PATH = "../figures/"
DATA_PATH = "../data/"

BAND_LIST = ["100-1a"]#, "100-1b", "100-2a", "100-2b", "100-3a", "100-3b", "100-4a", "100-4b"]
GLITCH_TYPES = ["short", "long", "slow"]

SAMPRATE = 180.3737 #Hz
SIGMA = 0.01 #random std for the white noise

FAST_PART = 0 # seconds
CUT_OFF = 20
NO_TRUST = 3 # samples

NSECS = 2 # used for matched filter, estimating template etc

CLASSIFY_SECS_FIRST = 0.1 # seconds, used for first iteration of classification
CLASSIFY_SECS = 2

PLOTS = True