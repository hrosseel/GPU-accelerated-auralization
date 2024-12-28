import numpy as np
import time
import csv
import logging

from benchmark import prepare_benchmark, benchmark

# Configure logging
logging.basicConfig(filename='benchmark.log', level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info('Starting benchmark...')

# Default parameters
fs = 48_000
num_input_frames = 10_000

# filter length
filter_lengths = (fs * np.array([0.1, 0.5, 1, 2, 5, 10, 20])).astype(int)

# number of channels
num_channels = 2 ** np.arange(8)  # 1 to 128 channels

# block sizes
block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Prepare the benchmark
bench_logs = []
for filter_length in filter_lengths:
    for num_channel in num_channels:
        for block_size in block_sizes:
            logging.info(f'Benchmark started for Filter Length: {filter_length}, Number of Channels: {num_channel}, Block Size: {block_size}')
            input_length = int(num_input_frames * block_size)
            pc, pc_gpu, signal_batch = prepare_benchmark(block_size, input_length, filter_length, num_channel)
            time.sleep(0.5)

            cpu_time = benchmark(pc, signal_batch)
            time.sleep(0.5)
            gpu_time = benchmark(pc_gpu, signal_batch)

            bench_logs.append([filter_length, num_channel, block_size, *cpu_time, *gpu_time])

            logging.info(f'Benchmark finished for Filter Length: {filter_length}, Number of Channels: {num_channel}, Block Size: {block_size}, CPU_mean: {cpu_time[0]}, CPU_median: {cpu_time[1]}, CPU_std_dev: {cpu_time[2]}, CPU_min: {cpu_time[3]}, CPU_max: {cpu_time[4]}, GPU_mean: {gpu_time[0]}, GPU_median: {gpu_time[1]}, GPU_std_dev: {gpu_time[2]}, GPU_min: {gpu_time[3]}, GPU_max: {gpu_time[4]}')

logging.info('Benchmark finished!')
logging.info('Writing results to CSV file...')

# Write the results to a CSV file
with open('benchmark_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filter Length', 'Number of Channels', 'Block Size', 'CPU Mean', 'CPU Median', 'CPU Std Dev', 'CPU Min', 'CPU Max', 'GPU Mean', 'GPU Median', 'GPU Std Dev', 'GPU Min', 'GPU Max'])
    writer.writerows(bench_logs)

logging.info('Results written to CSV file.')