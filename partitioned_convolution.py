import numpy as np

import torch
import os

# Set the TORCH_CUDA_ARCH_LIST environment variable
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
from torch.utils.cpp_extension import load_inline


# Load the CUDA code
# ================================================================
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src],
                       functions=funcs, extra_cuda_cflags=["-Xptxas -O3"]
                       if opt else [], verbose=verbose, name="inline_ext")


# Load CUDA code from file "cuda/kernel.cu"
cuda_code_path = os.path.join(os.path.dirname(__file__), "cuda/kernel.cu")
cuda_src = open(cuda_code_path, "r").read()

# Load CUDA code and function signature
cpp_src = ("torch::Tensor part_conv_gpu(torch::Tensor input_fd, torch::Tensor fdl, "
           "torch::Tensor filters_fd, int fdl_cursor, int K, int B, int C);")
module = load_cuda(cuda_src, cpp_src, ['part_conv_gpu'], verbose=False)
# ================================================================


# Main class
class PartitionedConvolution:

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
        remainder = self.K * self.B - self.FL
        self.filter_parts = np.pad(filter_td, ((0, 0), (0, remainder)),
                                   mode='constant').reshape(self.C, self.B, self.K, order='F')

        # Transform the filter to the frequency domain
        self.filters_fd = self.transform_rfft_filter(self.filter_parts)

        # Initialize the frequency-domain delay line (FDL)
        self.fdl = torch.zeros(
            (self.B + 1, self.K), dtype=torch.complex128)
        self.fdl_cursor = 0

        # Initialize the input buffer
        self.input_buffer_td = torch.zeros(2 * self.B, dtype=self.dtype)

    def convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform the partitioned convolution
        :param signal: The input signal (shape: (B,))
        :return: The output signal (shape: (C, B + 1))
        """
        # Validate the input signal
        if signal.shape != (self.B,):
            raise ValueError(
                "The input signal must be a 1D array with shape (B,)")

        # Put the incoming signal in the input buffer after sliding the previous signal
        self.input_buffer_td[:self.B] = self.input_buffer_td[self.B:]
        self.input_buffer_td[self.B:] = torch.tensor(signal)

        # Compute the RFFT of the signals (real-to-complex FFT)
        input_fd = torch.fft.rfft(self.input_buffer_td)  # shape: (B + 1)

        # Perform the actual convolution
        output_fd = self.perform_convolution(input_fd)

        self.fdl_cursor = (self.fdl_cursor + 1) % self.K  # Update the index

        # Perform the inverse RFFT to obtain the output signal
        output_td = torch.fft.irfft(output_fd, axis=1)  # shape: (C, 2 * B)

        # Only return the valid samples
        return output_td[:, self.B:].T

    def transform_rfft_filter(self, filters_td: np.ndarray) -> torch.Tensor:
        # Partition the filter into blocks of length B, and zero-pad another B samples
        filters_padded = np.pad(
            np.array(filters_td), ((0, 0), (0, self.B), (0, 0)), mode='constant')  # shape: (C, 2 * B, K)

        filters_padded = torch.tensor(filters_padded)
        # Compute the RFFT of the filters (real-to-complex FFT)
        return torch.fft.rfft(filters_padded, axis=1)  # shape: (K, B + 1, C)

    def perform_convolution(self, input_fd: torch.Tensor | np.ndarray) -> torch.Tensor:
        raise NotImplementedError(
            "This method is not implemented in this class.")


# CPU implementation
class PartitionedConvolutionCPU(PartitionedConvolution):

    def __init__(self, filter_td: torch.Tensor, block_length_samples: int, dtype: np.dtype = torch.float64):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (C, FL))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        PartitionedConvolution.__init__(
            self, filter_td, block_length_samples, dtype)

    def perform_convolution(self, input_fd: torch.Tensor | np.ndarray) -> torch.Tensor:

        if isinstance(input_fd, np.ndarray):
            input_fd = torch.tensor(input_fd)

        # Store the fd signal in a frequency-domain delay line
        self.fdl[:, self.fdl_cursor] = input_fd

        # Perform the complex multiplication between the fdl and the filter partitions
        output_fd = torch.zeros((self.C, self.B + 1), dtype=torch.complex128)
        for k in range(self.K):
            cursor = (self.fdl_cursor - k) % self.K
            output_fd += torch.mul(self.fdl[:, cursor],
                                   self.filters_fd[:, :, k])
        return output_fd


# GPU implementation
class PartitionedConvolutionGPU(PartitionedConvolution):

    def __init__(self, filter_td: torch.Tensor, block_length_samples: int, dtype: np.dtype = torch.float64):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (C, FL))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        PartitionedConvolution.__init__(
            self, filter_td, block_length_samples, dtype)

        # Load the filters to the GPU
        self.filters_fd_gpu = self.filters_fd.to(
            'cuda').type(torch.complex128).contiguous()
        # Load the FDL to the GPU
        self.fdl_gpu = self.fdl.to('cuda').type(torch.complex128).contiguous()

    def perform_convolution(self, input_fd: torch.Tensor) -> torch.Tensor:
        # Move the input spectrum to the GPU
        input_fd_gpu = input_fd.to('cuda').type(torch.complex128).contiguous()
        # Perform the convolution on the GPU
        output_fd = module.part_conv_gpu(
            input_fd_gpu, self.fdl_gpu, self.filters_fd_gpu, self.fdl_cursor, self.K, self.B, self.C)

        return output_fd.cpu()
