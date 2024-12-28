# File to benchmark the performance of the different algorithms
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import partitioned_convolution as fpc


def prepare_benchmark(block_size: int = 256, input_length: int = 48_000 * 1, filter_length: int = 48_000 * 10, num_channels: int = 24) -> list:

    # Create the filter
    filter_td = np.random.randn(filter_length * num_channels).astype(
        np.float32).reshape(num_channels, filter_length, order='C')

    # Create the partitioned convolution object
    pc = fpc.PartitionedConvolutionCPU(filter_td, block_size)
    pc_gpu = fpc.PartitionedConvolutionGPU(filter_td, block_size)

    # Define the signal batch
    signal = np.random.randn(input_length)
    pad_length = int(np.ceil(input_length / block_size) * block_size - input_length)  # Don't pad with additional zeros (filter) for benchmark consistency
    signal_batch = np.pad(signal, (0, pad_length), mode='constant').reshape(-1, block_size)

    return pc, pc_gpu, signal_batch


def benchmark(pc: fpc.PartitionedConvolution, signal_batch: np.ndarray, warmup_it: int = 10) -> list:
    # Warm-up
    for _ in range(warmup_it):
        pc.convolve(np.zeros(pc.B))

    log = []
    for input_idx, input_batch in enumerate(signal_batch):
        start_time = time.perf_counter()
        output = pc.convolve(input_batch)
        end_time = time.perf_counter()
        log.append(end_time - start_time)

    std_dev = np.std(log)
    mean = np.mean(log)
    median = np.median(log)
    min_time = np.min(log)
    max_time = np.max(log)

    return mean, median, std_dev, min_time, max_time


# Benchmark the GPU implementation

# # Warm-up
# for _ in range(warmup_it):
#     pc_gpu.convolve(np.zeros(B))

# log_gpu = []
# for input_idx, input_batch in enumerate(signal_batch):
#     start_time = time.perf_counter()
#     output = pc_gpu.convolve(input_batch)
#     end_time = time.perf_counter()
#     log_gpu.append(end_time - start_time)

# print(f"GPU time: {end_time - start_time} s")
# print(f"  Mean iteration: {np.mean(log_gpu)} s")
# print(f"  Median iteration: {np.median(log_gpu)} s")
# print(f"  Standard deviation: {np.std(log_gpu)} s")
# print(f"  Fastest iteration: {np.min(log_gpu)} s")
# print(f"  Slowest iteration: {np.max(log_gpu)} s")

# # Plot the results
# plt.figure()
# plt.plot(log_cpu, label='CPU')
# plt.plot(log_gpu, label='GPU')
# plt.legend()
# plt.xlabel('Iteration')
# plt.ylabel('Time (s)')
# plt.title('Partitioned convolution benchmark')
# plt.show()
