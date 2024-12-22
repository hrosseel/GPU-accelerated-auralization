# File to benchmark the performance of the different algorithms
import time
import numpy as np
import torch

import matplotlib.pyplot as plt

import partitioned_convolution as fpc


# parameters
warmup_it = 10


# Define the parameters
B = 512
C = 24
FL = 300_000
input_len = B * 10

# Create the filter
filter_td = np.random.randn(C * FL).astype(np.float32)
filter_td = filter_td.reshape(C, FL, order='C')

# Create the partitioned convolution object
pc = fpc.PartitionedConvolutionCPU(filter_td, B)
pc_gpu = fpc.PartitionedConvolutionGPU(filter_td, B)

# Define the signal batch
signal = np.random.randn(input_len)
signal_batch = np.pad(signal, (0, int(np.ceil(FL / B) * B)),
                      mode='constant').reshape(-1, B)

# Benchmark the CPU implementation

# Warm-up
for _ in range(warmup_it):
    pc.convolve(np.zeros(B))

log_cpu = []
for input_idx, input_batch in enumerate(signal_batch):
    start_time = time.perf_counter()
    output = pc.convolve(input_batch)
    end_time = time.perf_counter()
    log_cpu.append(end_time - start_time)

print(f"CPU time: {end_time - start_time} s")
print(f"  Mean iteration: {np.mean(log_cpu)} s")
print(f"  Median iteration: {np.median(log_cpu)} s")
print(f"  Fastest iteration: {np.min(log_cpu)} s")
print(f"  Slowest iteration: {np.max(log_cpu)} s")

# Benchmark the GPU implementation

# Warm-up
for _ in range(warmup_it):
    pc_gpu.convolve(np.zeros(B))

log_gpu = []
for input_idx, input_batch in enumerate(signal_batch):
    start_time = time.perf_counter()
    output = pc_gpu.convolve(input_batch)
    end_time = time.perf_counter()
    log_gpu.append(end_time - start_time)

print(f"GPU time: {end_time - start_time} s")
print(f"  Mean iteration: {np.mean(log_gpu)} s")
print(f"  Median iteration: {np.median(log_gpu)} s")
print(f"  Fastest iteration: {np.min(log_gpu)} s")
print(f"  Slowest iteration: {np.max(log_gpu)} s")


# Plot the results
plt.figure()
plt.plot(log_cpu, label='CPU')
plt.plot(log_gpu, label='GPU')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.title('Partitioned convolution benchmark')
plt.show()