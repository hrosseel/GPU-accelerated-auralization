"""
Partitioned Convolution

This module implements a partitioned convolution algorithm for real-time systems. The algorithm
is based on the partitioned convolution theorem, which states that the convolution of two signals
can be computed by partitioning the signals into blocks and performing the convolution in the
frequency domain, which allows for efficient computation of the convolution by exploiting the
parallel structure of the algorithm.
"""
import os

from numba import njit, prange, complex64, int64
import numpy as np
import torch

# Set the TORCH_CUDA_ARCH_LIST environment variable
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

from torch.utils.cpp_extension import load_inline


# Load the CUDA code
# ================================================================
def load_cuda(cuda_src, cpp_src, funcs, extra_cuda_cflags=[], verbose=False):
    random_id = np.random.randint(0, 1e6)
    """
    Load the CUDA code
    """
    return load_inline(
        cuda_sources=[cuda_src],
        cpp_sources=[cpp_src],
        functions=funcs,
        extra_cuda_cflags=["-O2"] + extra_cuda_cflags,
        verbose=verbose,
        name=f"inline_ext_{random_id}"
    )

# Load CUDA code from file "cuda/kernel.cu"
cuda_code_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
cuda_src = open(cuda_code_path, "r").read()
cpp_src = "torch::Tensor part_conv_gpu(torch::Tensor input_fd, torch::Tensor fdl, torch::Tensor filters_fd, int fdl_cursor);"
# ================================================================

# Multi-threaded CPU implementation of the complex multiplication
# ================================================================
@njit(complex64[:, ::1](complex64[:, :, :], complex64[:, :, ::1], int64, complex64[:, :, ::1]), parallel=True)
def cpu_multiply(filters_fd: np.ndarray, fdl: np.ndarray, fdl_cursor: int, temp_buffer: np.ndarray) -> np.ndarray:
    filter_channels, _, K = filters_fd.shape
    for k in prange(K):
        cursor = (K + (fdl_cursor - k)) % K
        temp_buffer[k, :, :] = filters_fd[:, :, k] * fdl[:, :, cursor]
    return temp_buffer.sum(axis=0)


@njit(complex64[:, ::1](complex64[:, :, :], complex64[:, :, ::1], int64, complex64[:, :, ::1]), parallel=True)
def cpu_multiply_single_input(filters_fd: np.ndarray, fdl: np.ndarray, fdl_cursor: int, temp_buffer: np.ndarray) -> np.ndarray:
    K = filters_fd.shape[2]
    for k in prange(K):
        cursor = (K + (fdl_cursor - k)) % K
        temp_buffer[k, :, :] = filters_fd[:, :, k] * fdl[0, :, cursor]
    return temp_buffer.sum(axis=0)
# ================================================================

# Main class
class PartitionedConvolution:

    def __init__(self, filter_td: torch.Tensor, block_length_samples: int,
                 num_input_channels: int = 1):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (C, FL))
        :param block_length_samples: The block length B
        :param num_input_channels: The number of input channels (default: 1)
        :param dtype: The data type
        """
        if filter_td.ndim != 2:
            raise ValueError(
                "The filter must be a 2D array with shape (num_channels, filter_length).")

        self.C, self.FL = filter_td.shape
        self.input_C = num_input_channels
        self.B = block_length_samples
        self.K = int(np.ceil(self.FL / self.B))

        # Validate if FL > B
        if self.FL < self.B:
            raise ValueError(
                "The filter length must be greater than the block length.")
        # validate block length
        if self.B < 1:
            raise ValueError("The block length must be greater than 1.")
        # Validate the filter length
        if self.FL < 1:
            raise ValueError("The filter length must be greater than 1.")
        # Validate the number of channels
        if self.C < 1:
            raise ValueError("The number of channels must be greater than 1.")
        if self.input_C not in [1, self.C]:
            raise ValueError("The number of input channels must be 1 or equal to the number of filter channels.")

        # Create the filter blocks
        self.filters_fd = self.__create_filter_blocks__(filter_td)
        # Initialize the frequency-domain delay line (FDL)
        self.fdl = torch.zeros((self.input_C, self.B + 1, self.K), dtype=torch.complex64)
        self.fdl_cursor = 0

        # Initialize the input buffers
        self.input_buffer_td = torch.zeros(self.input_C, 2 * self.B)

    def __parse_input__(self, signal: np.ndarray) -> None:
        # Validate the input signal
        if signal.shape != (self.input_C, self.B):
            if signal.shape == (self.B,) and self.input_C == 1:
                signal = signal.reshape(1, -1)
            else:
                raise ValueError(f"The input signal must be a 2D array with shape ({self.input_C}, {self.B}).")
        
        if signal.dtype != torch.float32:
            if signal.dtype == np.float32:
                signal = torch.tensor(signal)
            else:
                raise ValueError("The input signal must be of type float32.")

        # Put the incoming signal in the input buffer after sliding the previous signal
        self.input_buffer_td[:, :self.B] = self.input_buffer_td[:, self.B:]
        self.input_buffer_td[:, self.B:] = signal

        # Compute the RFFT of the signals (real-to-complex FFT)
        input_fd = torch.fft.rfft(self.input_buffer_td, dim=1)  # shape: (input_C, B + 1)
        return input_fd

    def convolve(self, signal_td: np.ndarray) -> np.ndarray:
        """
        Perform the partitioned convolution
        :param signal: The input signal (shape: (C, B))
        :return: The output signal (shape: (C, B + 1))
        """
        # Transform the input signal to the frequency domain
        signal_fd = self.__parse_input__(signal_td)

        # Perform the actual convolution
        output_fd = self.__perform_convolution__(signal_fd)
        self.fdl_cursor = (self.fdl_cursor + 1) % self.K  # Update the fdl_cursor

        # Perform the inverse RFFT to obtain the output signal
        output_td = torch.fft.irfft(output_fd, axis=1)  # shape: (C, 2 * B)

        # Only return the valid samples
        return output_td[:, self.B:]

    def __create_filter_blocks__(self, filter_td: np.ndarray) -> torch.Tensor:
        # create filter partitions
        remainder = self.K * self.B - self.FL
        filter_parts = np.pad(filter_td, ((0, 0), (0, remainder)), mode='constant'
                              ).reshape((self.C, self.B, self.K), order='F')

        # Partition the filter into blocks of length B, and zero-pad another B samples
        filters_padded = np.pad(
            np.array(filter_parts), ((0, 0), (0, self.B), (0, 0)), mode='constant')  # shape: (C, 2 * B, K)

        # Compute the RFFT of the filters (real-to-complex FFT)
        # Note: torch.fft.rfft messes up the ordering (F-contiguous) of the array
        return torch.from_numpy(np.fft.rfft(filters_padded, axis=1).astype(np.complex64))  # shape: (K, B + 1, C)
        

    def __perform_convolution__(self, input_fd: torch.Tensor | np.ndarray) -> torch.Tensor:
        raise NotImplementedError(
            "This method is not implemented in this class.")


# CPU implementation
class PartitionedConvolutionCPU(PartitionedConvolution):
    """
    Partitioned Convolution implemented on the CPU
    """
    def __init__(self, filter_td: torch.Tensor, block_length_samples: int, num_input_channels: int = 1):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (C, FL))
        :param block_length_samples: The block length B
        :param num_input_channels: The number of input channels (default: 1)
        """
        PartitionedConvolution.__init__(self, filter_td, block_length_samples, num_input_channels=num_input_channels)
        self.temp_buffer = np.empty((self.K, self.C, self.B + 1), dtype=np.complex64)

        # Convert to numpy array
        self.fdl = self.fdl.numpy()
        self.filters_fd = self.filters_fd.numpy()


    def __perform_convolution__(self, input_fd: torch.Tensor | np.ndarray) -> torch.Tensor:

        if isinstance(input_fd, torch.Tensor):
            input_fd = input_fd.numpy().astype(np.complex64)

        # Store the fd signal in a frequency-domain delay line
        self.fdl[:, :, self.fdl_cursor] = input_fd
        
        # Perform the complex multiplication between the fdl and the filter partitions
        if self.input_C == 1:
            output_fd = cpu_multiply_single_input(self.filters_fd, self.fdl, self.fdl_cursor, self.temp_buffer)
        else:
            output_fd = cpu_multiply(self.filters_fd, self.fdl, self.fdl_cursor, self.temp_buffer)
        return torch.from_numpy(output_fd)


# GPU implementation
class PartitionedConvolutionGPU(PartitionedConvolution):
    """
    Partitioned Convolution implemented on the GPU
    """
    def __init__(self, filter_td: torch.Tensor, block_length_samples: int, num_input_channels: int = 1):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (C, FL))
        :param block_length_samples: The block length B
        :param num_input_channels: The number of input channels (default: 1)
        """
        PartitionedConvolution.__init__(self, filter_td, block_length_samples, num_input_channels=num_input_channels)

        # Compile CUDA code with specific flags for better performance
        extra_cuda_cflags = [f"-DNUM_CHANNELS={self.C}", f"-DBLOCK_SIZE={self.B}", f"-DNUM_PARTS={self.K}"]
        if self.input_C == self.C:
            extra_cuda_cflags.append("-DMULTI_INPUT")  # Use multi-input mode

        self.module = load_cuda(cuda_src, cpp_src, ['part_conv_gpu'], extra_cuda_cflags)

        # Load the filters to the GPU
        self.filters_fd_gpu = self.filters_fd.to('cuda').type(torch.complex64).contiguous()
        # Load the FDL to the GPU
        self.fdl_gpu = self.fdl.to('cuda').type(torch.complex64).contiguous()

    def __perform_convolution__(self, input_fd: torch.Tensor) -> torch.Tensor:
        # Move the input spectrum to the GPU
        input_fd_gpu = input_fd.to('cuda').type(torch.complex64).contiguous()
        # Perform the convolution on the GPU
        output_fd = self.module.part_conv_gpu(input_fd_gpu, self.fdl_gpu, self.filters_fd_gpu, self.fdl_cursor)
        return output_fd.cpu()  # Move the output back to the CPU
