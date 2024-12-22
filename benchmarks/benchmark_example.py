# benchmark.py
import pyperf
import numpy as np
import partitioned_convolution as pc

def create_filters(C, FL):
    return np.random.randn(C, FL).astype(np.float32)

def create_input_frames(B, FL, input_len):
    signal = np.random.randn(input_len).astype(np.float32)
    return np.pad(signal, (0, int(np.ceil(FL / B) * B)), mode='constant').reshape(-1, B)

def bencher(loops, part_conv: pc.PartitionedConvolution, input_frames) -> float:
    t0 = pyperf.perf_counter()
    [part_conv.convolve(frame) for frame in input_frames]
    return pyperf.perf_counter() - t0

def main():
    runner = pyperf.Runner(min_time=0.01)

    # Define the parameters
    B = 256
    C = 24
    FL = 10_000
    input_len = B * 1

    # Create the filter
    filter_td = create_filters(C, FL)
    # Create input frames
    input_frames = create_input_frames(B, FL, input_len)

    # Create the partitioned convolution object
    pc_cpu = pc.PartitionedConvolutionCPU(filter_td, B)
    pc_gpu = pc.PartitionedConvolutionGPU(filter_td, B)

    result_cpu = runner.bench_time_func('CPU - benchmark', bencher, pc_cpu, input_frames)
    result_gpu = runner.bench_time_func('GPU - benchmark', bencher, pc_gpu, input_frames)

if __name__ == "__main__":
    main()
