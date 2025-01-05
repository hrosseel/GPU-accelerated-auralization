import matplotlib.pyplot as plt
import numpy as np
import csv
import os

workspace_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_filter_len_path = os.path.join(workspace_dir, '../benchmark_filter_len.npz')

# Default parameters
fs = 48_000
num_input_frames = 10_000
default_num_channels = 32
default_block_size = 128  # 2.6 ms latency budget
default_filter_length = fs * 10  # 10 seconds

# Load the benchmark results
benchmark_filter_len = np.load(benchmark_filter_len_path, allow_pickle=True)

# Plot the benchmark results for different filter lengths (Box plot)
# =================================================================

# Extract the data -- filter lengths, CPU logs, and GPU logs
filter_lengths = benchmark_filter_len['filter_lengths'] / fs
cpu_logs = benchmark_filter_len['cpu_logs']
gpu_logs = benchmark_filter_len['gpu_logs']

# Plot the results with uncertainty bounds  
plt.figure()
plt.plot(filter_lengths, np.median(cpu_logs, axis=1), label='CPU', color='blue')
plt.fill_between(filter_lengths, cpu_logs.min(axis=1), cpu_logs.max(axis=1), color='blue', alpha=0.1)
plt.plot(filter_lengths, np.median(gpu_logs, axis=1), label='GPU', color='red')
plt.fill_between(filter_lengths, gpu_logs.min(axis=1), gpu_logs.max(axis=1), alpha=0.1, color='red')

plt.hlines(default_block_size / fs, xmin=0, xmax=20, label="Latency Budget: $B = 256$", linestyles='dashed', color='black')

# plot log-log scale
plt.yscale('log')

plt.legend()
plt.ylabel('Time (seconds)')
plt.grid()
