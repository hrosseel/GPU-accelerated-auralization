import numpy as np

import torch
import os

from torch.utils.cpp_extension import load_inline


def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs, extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")


# Load CUDA code from file "cuda/kernel.cu"
cuda_code_path = os.path.join(os.path.dirname(__file__), "cuda/kernel.cu")
cuda_src = open(cuda_code_path, "r").read()

# Load CUDA code and function signature
cpp_src = "torch::Tensor part_conv_gpu(torch::Tensor input_fd, torch::Tensor fdl, torch::Tensor filters_fd, int fdl_cursor, int K, int B, int C);"

module = load_cuda(cuda_src, cpp_src, ['part_conv_gpu'], verbose=False)


class PartitionedConvolutionGPU:

    def __init__(self, filter_td: torch.Tensor, block_length_samples: int, dtype: np.dtype = torch.float64):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (C, FL))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        if filter_td.ndim != 2:
            raise ValueError(
                "The filter must be a 2D array with shape (num_channels, filter_length).")

        self.C, self.FL = filter_td.shape
        self.B = block_length_samples
        self.dtype = dtype

        # Validate if FL > B
        if self.FL < self.B:
            raise ValueError(
                "The filter length must be greater than the block length.")
        # Validate the data type
        if self.dtype not in [torch.float32, torch.float64]:
            raise ValueError("The data type must be float32 or np.float64.")
        # validate block length
        if self.B < 1:
            raise ValueError("The block length must be greater than 1.")
        # Validate the filter length
        if self.FL < 1:
            raise ValueError("The filter length must be greater than 1.")
        # Validate the number of channels
        if self.C < 1:
            raise ValueError("The number of channels must be greater than 1.")

        # Calculate the number of partitions
        if self.FL % self.B != 0:
            self.K = self.FL // self.B + 1
        else:
            self.K = self.FL // self.B

        # create filter partitions
        self.filter_parts = np.pad(
            filter_td, ((0, 0), (0, self.K * self.B - self.FL)), mode='constant').reshape(self.C, self.B, self.K, order='F')

        assert (self.filter_parts[0, :self.B, 0]
                == filter_td[0, :self.B]).all()

        # Transform the filter to the frequency domain and load it to the GPU
        self.filters_fd = self.transform_rfft_filter(self.filter_parts)
        self.filters_fd_gpu = torch.tensor(
            self.filters_fd, dtype=torch.complex128, device='cuda').contiguous()

        # Initialize the frequency-domain delay line (FDL)
        self.fdl_gpu = torch.zeros(
            (self.B + 1, self.K), dtype=torch.complex128, device='cuda').contiguous()
        self.fdl_cursor = 0

        # Initialize the input buffer
        self.input_buffer_td = np.zeros(2 * self.B)

    def convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform the partitioned convolution
        :param signal: The input signal (shape: (B,))
        :return: The output signal (shape: (B, C))
        """
        # Validate the input signal
        if signal.shape != (self.B,):
            raise ValueError(
                "The input signal must be a 1D array with shape (B,).")

        # Put the incoming signal in the input buffer after sliding the previous signal
        self.input_buffer_td[:self.B] = self.input_buffer_td[self.B:]
        self.input_buffer_td[self.B:] = signal

        # Compute the RFFT of the signals (real-to-complex FFT)
        input_fd = np.fft.rfft(self.input_buffer_td)  # shape: (B + 1)

        # Move the input to the GPU
        input_fd_gpu = torch.tensor(
            input_fd, dtype=torch.complex128, device='cuda').contiguous()

        output_fd = module.part_conv_gpu(
            input_fd_gpu, self.fdl_gpu, self.filters_fd_gpu, self.fdl_cursor, self.K, self.B, self.C)

        self.fdl_cursor = (self.fdl_cursor + 1) % self.K  # Update the index

        # Perform the inverse RFFT to obtain the output signal
        output_td = np.fft.irfft(output_fd.cpu(), axis=1)  # shape: (C, 2 * B)

        # Only return the valid samples
        return output_td[:, self.B:].T

    def transform_rfft_filter(self, filters_td: np.ndarray) -> np.ndarray:
        # Partition the filter into blocks of length B, and zero-pad another B samples
        filters_padded = np.pad(
            filters_td, ((0, 0), (0, self.B), (0, 0)), mode='constant')  # shape: (C, 2 * B, K)

        # Compute the RFFT of the filters (real-to-complex FFT)
        return np.fft.rfft(filters_padded, axis=1)  # shape: (K, B + 1, C)
