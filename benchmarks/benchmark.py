# File to benchmark the performance of the different algorithms
import time
import numpy as np
import torch

import partitioned_convolution as fpc

# Define the parameters
B = 128
C = 2
FL = 500_000
input_len = B * 800

# Create the filterß
filter_td = np.random.randn(C, FL).astype(np.float32)

# Create the partitioned convolution object
pc = fpc.PartitionedConvolutionCPU(filter_td, B)
pc_gpu = fpc.PartitionedConvolutionGPU(filter_td, B)

# Define the signal batch
signal = np.random.randn(input_len)
signal_batch = np.pad(signal, (0, int(np.ceil(FL / B) * B)),
                      mode='constant').reshape(-1, B)

# Benchmark the CPU implementation
# log = []
# for input_idx, input_batch in enumerate(signal_batch):
#     start_time = time.time()
#     output = pc.convolve(input_batch)
#     end_time = time.time()
#     log.append(end_time - start_time)

# print(f"CPU time: {end_time - start_time} s")
# print(f"  Mean iteration: {np.mean(log)} s")
# print(f"  Median iteration: {np.median(log)} s")
# print(f"  Slowest iteration: {np.max(log)} s")
# print(f"  Fastest iteration: {np.min(log)} s")

# Benchmark the GPU implementation
log = []
for input_idx, input_batch in enumerate(signal_batch):
    start_time = time.time()
    output = pc_gpu.convolve(input_batch)
    end_time = time.time()
    log.append(end_time - start_time)

print(f"GPU time: {end_time - start_time} s")
print(f"  Mean iteration: {np.mean(log)} s")
print(f"  Median iteration: {np.median(log)} s")
print(f"  Slowest iteration: {np.max(log)} s")
print(f"  Fastest iteration: {np.min(log)} s")