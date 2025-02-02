"""
This file contains the configuration for the benchmarks.
"""

# Default parameters
FS = 48_000
NUM_INPUT_FRAMES = 10_000
DEFAULT_NUM_CHANNELS = 32
DEFAULT_BLOCK_SIZE = 128  # 2.6 ms latency budget
DEFAULT_AUR_FILTER_LENGTH = FS * 10  # 10 seconds
DEFAULT_FC_FILTER_LENGTH = FS * 1  # 1 second
