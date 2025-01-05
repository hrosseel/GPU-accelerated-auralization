import numpy as np
import pytest
from partitioned_convolution_multi import PartitionedConvolutionMultiCPU, PartitionedConvolutionMultiGPU

from scipy.signal import convolve


# Test case 1: Single channel test
def test_single_channel_cpu():
    B = 32
    C = 1
    FL = 100

    # Define the filter (F-contiguous)
    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')

    # Load the partitioned convolution object
    pc = PartitionedConvolutionMultiCPU(filters_td, B)

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


# Test case 1: Single channel test GPU
def test_single_channel_gpu():
    B = 32
    C = 1
    FL = 100

    # Define the filter (F-contiguous)
    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')

    # Load the partitioned convolution object
    pc = PartitionedConvolutionMultiGPU(filters_td, B)

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
def test_dual_channels_cpu():
    B = 32
    C = 2
    FL = 100
    K = np.ceil(FL / B).astype(int)

    # Define the filter (F-contiguous)
    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')

    # Load the partitioned convolution object
    pc = PartitionedConvolutionMultiCPU(filters_td, B)
    
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


# Test case 2: Multiple channels test GPU
def test_dual_channels_gpu():
    B = 32
    C = 2
    FL = 75
    K = np.ceil(FL / B).astype(int)

    # Define the filter (F-contiguous)
    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')

    # Load the partitioned convolution object
    pc = PartitionedConvolutionMultiGPU(filters_td, B)

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


# Test case 3: Multiple channels test
def test_multi_channels_cpu():
    B = 256
    C = 100
    FL = B * 100
    I = 10
    K = np.ceil(FL / B).astype(int)

    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')
    pc = PartitionedConvolutionMultiCPU(filters_td, B)

    # Define the signal batch
    signal = np.random.randn(B * I)
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
            atol=5e-4, rtol=5e-4)


def test_multi_channels_multi_input_cpu():
    B = 256
    C = 100
    FL = B * 100
    I = 10
    K = np.ceil(FL / B).astype(int)

    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')
    pc = PartitionedConvolutionMultiCPU(filters_td, B, num_input_channels=C)

    # Define the signal batch
    signal = np.random.randn(C, B * I)
    signal_batch = signal.reshape(C, I, B).swapaxes(0, 1)

    output_len = signal.shape[1] + FL - 1

    true_output = np.zeros((output_len + B, C))
    for c in range(C):
        true_output[:output_len, c] = convolve(
            signal[c, :], filters_td[c, :], mode='full')

    for input_idx, input_batch in enumerate(signal_batch):
        output = pc.convolve(input_batch)
        np.testing.assert_allclose(
            output, true_output[input_idx * B:(input_idx + 1) * B, :],
            atol=5e-4, rtol=5e-4)


# Test case 3: Multiple channels test GPU
def test_multi_channels_gpu():
    B = 256
    C = 100
    FL = B * 100
    I = 10
    K = np.ceil(FL / B).astype(int)

    # Define the filter (F-contiguous)
    filters_td = np.random.randn(C * FL).astype(np.float32)
    filters_td = filters_td.reshape(C, FL, order='C')

    # Load the partitioned convolution object
    pc = PartitionedConvolutionMultiGPU(filters_td, B)

    # Define the signal batch
    signal = np.random.randn(B * I)
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
            atol=5e-4, rtol=5e-4)


if __name__ == "__main__":
#    pytest.main()
    test_multi_channels_multi_input_cpu()
