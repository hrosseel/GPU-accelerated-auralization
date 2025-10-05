"""
This script benchmarks the partitioned convolution implementation for different
filter lengths, number of channels, and block sizes. The results are saved to
.npz files.
"""
import time
import logging

from config import FS, NUM_INPUT_FRAMES, DEFAULT_NUM_CHANNELS, DEFAULT_BLOCK_SIZE, \
    DEFAULT_AUR_FILTER_LENGTH

import numpy as np
import partitioned_convolution as fpc


def prepare_benchmark(block_size: int = 256, input_length: int = FS * 1,
                      filter_length: int = FS * 10, num_channels: int = 24) -> list:
    """ Prepare the benchmark for the partitioned convolution"""
    # Create the filter
    filter_td = np.random.randn(filter_length * num_channels).astype(
        np.float32).reshape(num_channels, filter_length, order='C')

    # Create the partitioned convolution object
    pc = fpc.PartitionedConvolutionCPU(filter_td, block_size)
    pc_gpu = fpc.PartitionedConvolutionGPU(filter_td, block_size)

    # Define the signal batch
    signal = np.random.randn(input_length).astype(np.float32)
    # Don't pad with additional zeros (filter) for benchmark consistency
    pad_length = int(np.ceil(input_length / block_size)
                     * block_size - input_length)
    signal_batch = np.pad(signal, (0, pad_length),
                          mode='constant').reshape(-1, block_size)

    return pc, pc_gpu, signal_batch


def benchmark(pc: fpc.PartitionedConvolution, signal_batch: np.ndarray,
              warmup_it: int = 10) -> list:
    """ Benchmark the partitioned convolution implementation"""
    for _ in range(warmup_it):
        pc.convolve(np.zeros(pc.B).astype(np.float32))

    log = []
    for input_batch in signal_batch:
        start_time = time.perf_counter()
        _ = pc.convolve(input_batch)
        end_time = time.perf_counter()
        log.append(end_time - start_time)
    return log


def start_benchmark(filter_length: int, num_channels: int, block_size: int) -> tuple:
    """
    Start the benchmark for the partitioned convolution
    :param filter_length: The filter length
    :param num_channels: The number of channels
    :param block_size: The block size
    :return: The benchmark results
    """
    logging.info('Benchmark started for Filter Length: %d, Number of Channels: %d, Block Size: %d',
                 filter_length, num_channels, block_size)
    input_length = int(NUM_INPUT_FRAMES * block_size)
    pc, pc_gpu, signal_batch = prepare_benchmark(
        block_size, input_length, filter_length, num_channels)
    time.sleep(0.5)
    _cpu_log = benchmark(pc, signal_batch)
    time.sleep(0.5)
    _gpu_log = benchmark(pc_gpu, signal_batch)
    return _cpu_log, _gpu_log


# Configure logging
logging.basicConfig(filename='benchmark_part_conv.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logging.info('Starting benchmark...')


def bench_conv_filter_length():
    """ Benchmark partitioned convolution for different filter lengths """
    filter_lengths = (FS * np.array([0.1, 0.5, 1, 2, 5, 10, 20])).astype(int)
    cpu_logs = []
    gpu_logs = []
    for filter_length in filter_lengths:
        cpu_log, gpu_log = start_benchmark(
            filter_length, DEFAULT_NUM_CHANNELS, DEFAULT_BLOCK_SIZE)
        cpu_logs.append(cpu_log)
        gpu_logs.append(gpu_log)

    # Save the results to a file
    np.savez('./benchmarks/data/benchmark_conv_filter_len.npz', cpu_logs=cpu_logs,
             gpu_logs=gpu_logs, filter_lengths=filter_lengths)


def bench_conv_num_channels():
    """ Benchmark partitioned convolution for different number of channels """
    num_channels = 2 ** np.arange(8)  # 1 to 128 channels
    cpu_logs = []
    gpu_logs = []
    for num_channel in num_channels:
        cpu_log, gpu_log = start_benchmark(
            DEFAULT_AUR_FILTER_LENGTH, num_channel, DEFAULT_BLOCK_SIZE)
        cpu_logs.append(cpu_log)
        gpu_logs.append(gpu_log)

    # Save the results to a file
    np.savez('./benchmarks/data/benchmark_conv_num_channels.npz', cpu_logs=cpu_logs,
             gpu_logs=gpu_logs, num_channels=num_channels)


def bench_conv_block_size():
    """ Benchmark partitioned convolution for different block sizes """
    block_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    cpu_logs = []
    gpu_logs = []
    for block_size in block_sizes:
        cpu_log, gpu_log = start_benchmark(DEFAULT_AUR_FILTER_LENGTH, DEFAULT_NUM_CHANNELS,
                                           block_size)
        cpu_logs.append(cpu_log)
        gpu_logs.append(gpu_log)

    # Save the results to a file
    np.savez('./benchmarks/data/benchmark_conv_block_size.npz', cpu_logs=cpu_logs,
             gpu_logs=gpu_logs, block_sizes=block_sizes)


if __name__ == '__main__':
    bench_conv_filter_length()
    bench_conv_num_channels()
    bench_conv_block_size()
