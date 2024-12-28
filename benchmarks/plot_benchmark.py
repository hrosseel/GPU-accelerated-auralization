import matplotlib.pyplot as plt
import numpy as np
import csv
import os

workspace_dir = os.path.dirname(os.path.abspath(__file__))
csv_filepath = os.path.join(workspace_dir, '../benchmark_results_1000runs.csv')
# Read the results from the CSV file
with open(csv_filepath, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)
    data = np.array(list(reader))

# Extract the data
filter_lengths = np.unique(data[:, 0].astype(int))
num_channels = np.unique(data[:, 1].astype(int))
block_sizes = np.unique(data[:, 2].astype(int))
cpu_mean = data[:, 3].astype(float)
cpu_median = data[:, 4].astype(float)
cpu_std_dev = data[:, 5].astype(float)
cpu_min = data[:, 6].astype(float)
cpu_max = data[:, 7].astype(float)
gpu_mean = data[:, 8].astype(float)
gpu_median = data[:, 9].astype(float)
gpu_std_dev = data[:, 10].astype(float)
gpu_min = data[:, 11].astype(float)
gpu_max = data[:, 12].astype(float)

# Default plot parameters
filt_filter_len = 960000
filt_num_ch = 32
filt_block_size = 128
filter_indices = np.where((data[:, 0].astype(int) == filt_filter_len) & (data[:, 2].astype(int) == filt_block_size))

# Plot the results
fig, axs = plt.subplots(1, 1)
axs.set_title('Benchmark Results')
axs.set_xlabel('Number of Channels')
axs.set_ylabel('Time (s)')

# Plot the CPU results
axs.plot(num_channels, cpu_median[filter_indices], label='CPU Median')
low_bound, upper_bound = cpu_median[filter_indices] + (-cpu_std_dev[filter_indices], cpu_std_dev[filter_indices])
axs.fill_between(num_channels, low_bound, upper_bound, alpha=0.1)

axs.plot(num_channels, gpu_median[filter_indices], label='GPU Median')
low_bound, upper_bound = gpu_median[filter_indices] + (-gpu_std_dev[filter_indices], gpu_std_dev[filter_indices])
axs.fill_between(num_channels, low_bound, upper_bound, alpha=0.1)

axs.hlines(filt_block_size / 48_000, num_channels[0], num_channels[-1], colors='r', linestyles='dashed', label='Latency budget')
axs.legend()
