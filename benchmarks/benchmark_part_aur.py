"""
Benchmark the partitioned auralization implementation for different filter lengths,
feedback cancelation filter lengths, number of channels, and block sizes. The results are saved to .npz files.
"""
import time
import logging

from config import FS, NUM_INPUT_FRAMES, DEFAULT_NUM_CHANNELS, DEFAULT_BLOCK_SIZE, DEFAULT_AUR_FILTER_LENGTH, DEFAULT_FC_FILTER_LENGTH

import numpy as np
import torch
from partitioned_auralization import PartitionedAuralization

def prepare_benchmark(block_size: int = 256, input_length: int = FS * 1,
                      aur_filter_length: int = FS * 10, fc_filter_length: int = FS,
                      num_channels: int = 24) -> list:
    # Create the auralization and feedback cancellation filters
    aur_filter_td = torch.randn(num_channels, aur_filter_length, dtype=torch.float32)
    fc_filter_td = torch.randn(num_channels, fc_filter_length, dtype=torch.float32)

    # Create the partitioned auralization object
    pa_cpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_size, device='cpu')
    pa_gpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_size, device='gpu')

    # Define the signal batch
    signal = np.random.randn(input_length).astype(np.float32)
    # Don't pad with additional zeros (filter) for benchmark consistency
    pad_length = int(np.ceil(input_length / block_size) * block_size - input_length)
    signal_batch = np.pad(signal, (0, pad_length), mode='constant').reshape(-1, block_size)

    return pa_cpu, pa_gpu, signal_batch

def benchmark(pa: PartitionedAuralization, signal_batch: np.ndarray, warmup_it: int = 10) -> list:
    # Warm-up
    for _ in range(warmup_it):
        pa.auralization(np.zeros(pa.B).astype(np.float32))

    log = []
    for input_batch in signal_batch:
        start_time = time.perf_counter()
        _ = pa.auralization(input_batch)
        end_time = time.perf_counter()
        log.append(end_time - start_time)
    return log

def start_benchmark(aur_filter_length: int, fc_filter_length: int, num_channels: int,
                    block_size: int) -> tuple:
    """
    Perform the benchmark for the partitioned auralization
    :param aur_filter_length: The auralization filter length
    :param fc_filter_length: The feedback cancelation filter length
    :param num_channels: The number of channels
    :param block_size: The block size
    :return: The benchmark results
    """
    logging.info('Benchmark started for auralization filter length: %d samples, feedback path length: %d samples, Number of Channels: %d, Block Size: %d',
                 aur_filter_length, fc_filter_length, num_channels, block_size)
    
    input_length = int(NUM_INPUT_FRAMES * block_size)
   
    pa_cpu, pa_gpu, signal_batch = prepare_benchmark(block_size, input_length, aur_filter_length, fc_filter_length, num_channels)
  
    time.sleep(0.5)
    _cpu_log = benchmark(pa_cpu, signal_batch)
    time.sleep(0.5)
    _gpu_log = benchmark(pa_gpu, signal_batch)
    return _cpu_log, _gpu_log

# Configure logging
logging.basicConfig(filename='benchmark_part_aur.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

logging.info('Starting benchmark...')


#  Benchmark partitioned auralization for different aur filter lengths
#  =================================================================
aur_filter_lengths = (FS * np.array([0.1, 0.5, 1, 2, 5, 10, 20])).astype(int)
cpu_logs = []
gpu_logs = []
for aur_filter_length in aur_filter_lengths:
    cpu_log, gpu_log = start_benchmark(aur_filter_length, DEFAULT_FC_FILTER_LENGTH, DEFAULT_NUM_CHANNELS, DEFAULT_BLOCK_SIZE)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)

# Save the results to a file
np.savez('./benchmarks/data/benchmark_aur_filter_len_aur.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs,
         aur_filter_lengths=aur_filter_lengths)
#  =================================================================

# Benchmark partitioned auralization for different fc filter lengths
#  =================================================================
fc_filter_lengths = (FS * np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1., 2., 5.])).astype(int)
cpu_logs = []
gpu_logs = []
for fc_filter_length in fc_filter_lengths:
    cpu_log, gpu_log = start_benchmark(DEFAULT_AUR_FILTER_LENGTH, fc_filter_length, DEFAULT_NUM_CHANNELS, DEFAULT_BLOCK_SIZE)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)

# Save the results to a file
np.savez('./benchmarks/data/benchmark_aur_filter_len_fc.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs,
          fc_filter_lengths=fc_filter_lengths)

#  Benchmark partitioned auralization for different number of channels
#  =================================================================
num_channels = 2 ** np.arange(8)  # 1 to 128 channels
cpu_logs = []
gpu_logs = []
for num_channel in num_channels:
    cpu_log, gpu_log = start_benchmark(DEFAULT_AUR_FILTER_LENGTH, DEFAULT_FC_FILTER_LENGTH, num_channel, DEFAULT_BLOCK_SIZE)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)

# Save the results to a file
np.savez('./benchmarks/data/benchmark_aur_num_channels.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs,
         num_channels=num_channels)
#  =================================================================

#  Benchmark partitioned auralization for different block sizes
#  =================================================================
block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
cpu_logs = []
gpu_logs = []
for block_size in block_sizes:
    cpu_log, gpu_log = start_benchmark(DEFAULT_AUR_FILTER_LENGTH, DEFAULT_FC_FILTER_LENGTH, DEFAULT_NUM_CHANNELS, block_size)
    cpu_logs.append(cpu_log)
    gpu_logs.append(gpu_log)

# Save the results to a file
np.savez('./benchmarks/data/benchmark_aur_block_size.npz', cpu_logs=cpu_logs, gpu_logs=gpu_logs,
         block_sizes=block_sizes)
#  =================================================================
