# File to benchmark the performance of the different algorithms

import time
import numpy as np
from partitioned_convolution import PartitionedConvolutionCPU, PartitionedConvolutionGPU
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
start_time = time.time()
for input_idx, input_batch in enumerate(signal_batch):
    output = pc_gpu.convolve(input_batch).cpu()
end_time = time.time()
print(f"GPU time: {end_time - start_time} s")
print(
    f"GPU time per iteration: {(end_time - start_time) / (input_len // B)} s")

print(
    f"Latency in samples: {(end_time - start_time) / (input_len // B) * 48000}")
