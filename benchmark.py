# File to benchmark the performance of the different algorithms

import time
import numpy as np
import torch
from partitioned_convolution import PartitionedConvolutionCPU, PartitionedConvolutionGPU


# def setup(B, C, filter_len):
#     filter_td = np.random.randn(C, filter_len).astype(np.float32)
#     pc = PartitionedConvolutionCPU(filter_td, B)
#     pc_gpu = PartitionedConvolutionGPU(filter_td, B)
#     signal = np.random.randn(input_len)
#     signal_batch = np.pad(signal, (0, int(np.ceil(filter_len / B) * B)),
#                           mode='constant').reshape(-1, B)
#     return pc, pc_gpu, signal_batch


# Define the parameters
B = 256
C = 24
FL = 500_000
input_len = B * 1000

# Create the filter
filter_td = np.random.randn(C, FL).astype(np.float32)

# Create the partitioned convolution object
pc = PartitionedConvolutionCPU(filter_td, B)
pc_gpu = PartitionedConvolutionGPU(filter_td, B)

# Define the signal batch
signal = np.random.randn(input_len)
signal_batch = np.pad(signal, (0, int(np.ceil(FL / B) * B)),
                      mode='constant').reshape(-1, B)

# # Benchmark the CPU implementation
# start_time = time.time()
# for input_idx, input_batch in enumerate(signal_batch):
#     output = pc.convolve(input_batch)
# end_time = time.time()
# print(f"CPU time: {end_time - start_time} s")
# print(
#     f"CPU time per iteration: {(end_time - start_time) / (input_len // B)} s")

# Benchmark the GPU implementation
log = []
for input_idx, input_batch in enumerate(signal_batch):
    start_time = time.time()
    output = pc_gpu.convolve(input_batch)
    end_time = time.time()
    log.append(end_time - start_time)

print(f"Mean iteration: {np.mean(log)} s")
print(f"Median iteration: {np.median(log)} s")
print(f"Slowest iteration: {np.max(log)} s")
print(f"Fastest iteration: {np.min(log)} s")
