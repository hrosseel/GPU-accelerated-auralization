import numpy as np
import time
import csv
import logging

from benchmark import prepare_benchmark, benchmark

def perform_benchmark(filter_length: int, num_channels: int, block_size: int) -> tuple:
    """
    Perform the benchmark for the partitioned convolution
    :param filter_length: The filter length
    :param num_channels: The number of channels
    :param block_size: The block size
    :return: The benchmark results
    """
    logging.info(f'Benchmark started for Filter Length: {filter_length}, Number of Channels: {num_channels}, Block Size: {block_size}')
    input_length = int(num_input_frames * block_size)
    pc, pc_gpu, signal_batch = prepare_benchmark(block_size, input_length, filter_length, num_channels)
    time.sleep(0.2)
    cpu_log = benchmark(pc, signal_batch)
    time.sleep(0.2)
    gpu_log = benchmark(pc_gpu, signal_batch)
    return cpu_log, gpu_log


# Configure logging
logging.basicConfig(filename='benchmark_part_conv.log', level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info('Starting benchmark...')

# Default parameters
fs = 48_000
num_input_frames = 10_000
default_num_channels = 32
default_block_size = 128  # 2.6 ms latency budget
default_filter_length = fs * 10  # 10 seconds

#  Benchmark partitioned convolution for different filter lengths
#  =================================================================
filter_lengths = (fs * np.array([0.1, 0.5, 1, 2, 5, 10, 20])).astype(int)
cpu_logs = []
gpu_logs = []
for filter_length in filter_lengths:
    cpu_log, gpu_log = perform_benchmark(filter_length, default_num_channels, default_block_size)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)
    logging.info(f'Filter Length: {filter_length}, CPU Time: {cpu_logs[-1]}, GPU Time: {gpu_logs[-1]}')

# Save the results to a file
np.savez('benchmark_filter_len.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs, filter_lengths=filter_lengths)
#  =================================================================

#  Benchmark partitioned convolution for different number of channels
#  =================================================================
num_channels = 2 ** np.arange(8)  # 1 to 128 channels
cpu_logs = []
gpu_logs = []
for num_channel in num_channels:
    cpu_log, gpu_log = perform_benchmark(default_filter_length, num_channel, default_block_size)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)
    logging.info(f'Number of Channels: {num_channel}, CPU Time: {cpu_logs[-1]}, GPU Time: {gpu_logs[-1]}')

# Save the results to a file
np.savez('benchmark_num_channels.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs, num_channels=num_channels)
#  =================================================================

#  Benchmark partitioned convolution for different block sizes
#  =================================================================
block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
cpu_logs = []
gpu_logs = []
for block_size in block_sizes:
    cpu_log, gpu_log = perform_benchmark(default_filter_length, default_num_channels, block_size)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)
    logging.info(f'Block Size: {block_size}, CPU Time: {cpu_logs[-1]}, GPU Time: {gpu_logs[-1]}')

# Save the results to a file
np.savez('benchmark_block_size.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs, block_sizes=block_sizes)
#  =================================================================
