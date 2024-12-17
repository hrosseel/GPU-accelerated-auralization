import numpy as np
import pytest
from partitioned_convolution import PartitionedConvolution
from partitioned_convolution_gpu import PartitionedConvolutionGPU

from scipy.signal import convolve


# Test case 1: Single channel test
def test_single_channel():
    B = 32
    C = 1
    FL = 100

    filter_td = np.random.randn(FL, C).astype(np.float32)
    pc = PartitionedConvolution(filter_td, B)

    # Define the signal batch
    signal = np.random.randn(B * 10)
    signal_batch = np.pad(signal, (0, int(np.ceil(FL / B) * B)),
                          mode='constant').reshape(-1, B)

    true_output = convolve(signal, filter_td[:, 0], mode='full')
    true_output = np.pad(true_output, (0, B), mode='constant')

    for input_idx, input_batch in enumerate(signal_batch):
        output = pc.convolve(input_batch)[:, 0]
        np.testing.assert_allclose(
            output, true_output[input_idx * B:(input_idx + 1) * B], atol=1e-5, rtol=1e-5)


# Test case 1: Single channel test GPU
def test_single_channel_gpu():
    B = 32
    C = 1
    FL = 100

    filters_td = np.random.randn(C, FL).astype(np.float32)
    pc = PartitionedConvolutionGPU(filters_td, B)

    # Define the signal batch
    signal = np.random.randn(B * 10)
    signal_batch = np.pad(signal, (0, int(np.ceil(FL / B) * B)),
                          mode='constant').reshape(-1, B)

    true_output = convolve(signal, filters_td[0, :], mode='full')
    true_output = np.pad(true_output, (0, B), mode='constant')

    for input_idx, input_batch in enumerate(signal_batch):
        output = pc.convolve(input_batch)[:, 0]
        np.testing.assert_allclose(
            output, true_output[input_idx * B:(input_idx + 1) * B], atol=1e-5, rtol=1e-5)


# Test case 2: Multiple channels test
def test_dual_channels():
    B = 32
    C = 2
    FL = 64
    K = np.ceil(FL / B).astype(int)

    filter_td = np.random.randn(FL, C).astype(np.float32)
    pc = PartitionedConvolution(filter_td, B)

    # Define the signal batch
    signal = np.random.randn(B * 10)
    signal_batch = np.pad(signal, (0, int(K * B)),
                          mode='constant').reshape(-1, B)

    output_len = len(signal) + FL - 1

    true_output = np.zeros((output_len + B, C))
    for c in range(C):
        true_output[:output_len, c] = convolve(
            signal, filter_td[:, c], mode='full')

    for input_idx, input_batch in enumerate(signal_batch):
        output = pc.convolve(input_batch)
        np.testing.assert_allclose(
            output, true_output[input_idx * B:(input_idx + 1) * B],
            atol=1e-5, rtol=1e-5)


# Test case 2: Multiple channels test GPU
def test_dual_channels_gpu():
    B = 32
    C = 2
    FL = 64
    K = np.ceil(FL / B).astype(int)

    filters_td = np.random.randn(C, FL).astype(np.float32)
    pc = PartitionedConvolutionGPU(filters_td, B)

    # Define the signal batch
    signal = np.random.randn(B * 10)
    signal_batch = np.pad(signal, (0, int(K * B)),
                          mode='constant').reshape(-1, B)

    output_len = len(signal) + FL - 1

    true_output = np.zeros((output_len + B, C))
    for c in range(C):
        true_output[:output_len, c] = convolve(
            signal, filters_td[c, :], mode='full')

    for input_idx, input_batch in enumerate(signal_batch):
        output = pc.convolve(input_batch)
        np.testing.assert_allclose(
            output, true_output[input_idx * B:(input_idx + 1) * B],
            atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
