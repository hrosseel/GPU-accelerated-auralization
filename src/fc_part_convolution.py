from numba import njit, prange, complex64, int64
import numpy as np
import torch
import os


def get_num_partitions(filter_length: int, block_length: int) -> int:
    """
    Calculate the number of partitions required for the partitioned convolution
    :param filter_length: The filter length
    :param block_length: The block length
    :return: The number of partitions
    """
    if filter_length % block_length != 0:
        K = filter_length // block_length + 1
    else:
        K = filter_length // block_length
    return K


def create_filter_blocks(filters_td: np.ndarray, num_partitions: int, block_length: int) -> torch.Tensor:
    num_channels, filter_len = filters_td.shape
    remainder = num_partitions * block_length - filter_len
    filter_parts = np.pad(filters_td, ((0, 0), (0, remainder)), mode='constant').reshape(
        num_channels, block_length, num_partitions, order='F')

    # Partition the filter into blocks of length B, and zero-pad another B samples
    filters_padded = np.pad(
        np.array(filter_parts), ((0, 0), (0, block_length), (0, 0)), mode='constant')  # shape: (C, 2 * B, K)

    # Compute the RFFT of the filters (real-to-complex FFT)
    # Note: torch.fft.rfft messes up the ordering (F-contiguous) of the array
    # shape: (K, B + 1, C)
    return torch.from_numpy(np.fft.rfft(filters_padded, axis=1))


# Multi-threaded CPU implementation of the complex multiplication
# ================================================================
@njit(complex64[:, ::1](complex64[:, :, :], complex64[:, ::1], int64, complex64[:, :, :],
                        complex64[:, ::1], int64, int64, complex64[:, :, ::1]), parallel=True)
def cpu_aur_multiply(aur_filters_fd: np.ndarray, aur_fdl: np.ndarray, aur_fdl_cursor: int, fc_filters_fd: np.ndarray,
                     fc_fdl: np.ndarray, fc_fdl_cursor: int, K_aur: int, temp_buffer: np.ndarray) -> np.ndarray:
    for k in prange(K_aur):
        aur_cursor = (aur_fdl_cursor - k) % K_aur
        
        for c_idx, filter_fd in enumerate(filters_fd):
            temp_buffer[k, c_idx] = filter_fd[:, k] * fdl[:, cursor]
    
    return temp_buffer.sum(axis=0)
# ================================================================


# Main class
class PartitionedAuralization:
    """
    Partitioned Auralization class

    This class implements a partitioned algorithm for real-time auralization system. The algorithm consists of two main
    components: the auralization filter and the feedback cancellator. The auralization filter is used to simulate the room
    acoustics, while the feedback cancellator is used to cancel the feedback signal. The algorithm is partitioned into
    blocks of length B, and the convolution and subtraction of the feedback signal are performed in the frequency domain.

    Currently, only feedback-cancelation of a single input channel is supported (i.e., SIMO auralization systems). The algorithm
    can be extended to support multiple input channels (MIMO auralization systems) by extending the feedback cancellator to
    support multiple input channels. For each input channel, a total of C feedback cancellators are required (where C is the
    number of output channels).
    """

    def __init__(self, aur_filter_td: torch.Tensor, fc_filter_td: torch.Tensor, block_length_samples: int, dtype: np.dtype = torch.float64):
        """
        Initialize the partitioned auralization class
        :param aur_filter_td: The auralization filter in the time domain (shape: (C, FL_AUR))
        :param fc_filter_td: The feedback cancelation filter in the time domain (shape: (L, FL_FC))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        if aur_filter_td.ndim != 2:
            raise ValueError(
                "The filter must be a 2D array with shape (num_channels, filter_length).")

        self.C, self.FL_AUR = aur_filter_td.shape
        L, self.FL_FC = fc_filter_td.shape
        self.B = block_length_samples
        self.dtype = dtype

        # Validate if C == L
        if self.C != L:
            raise ValueError(
                "The number of channels in the auralization filter must be equal to the number of cancelation filters.")
        # Validate if FL_AUR > B
        if self.FL_AUR < self.B:
            raise ValueError(
                "The filter length must be greater than the block length.")
        # Validate if FL_FC > B
        if self.FL_FC < self.B:
            raise ValueError(
                "The feedback cancelation filterlength must be greater than the block length.")
        # Validate the data type
        if self.dtype not in [torch.float32, torch.float64]:
            raise ValueError("The data type must be float32 or np.float64.")
        # validate block length
        if self.B < 1:
            raise ValueError("The block length must be greater than 1.")
        # Validate the filter length
        if self.FL_AUR < 1:
            raise ValueError("The filter length must be greater than 1.")
        # Validate the number of channels
        if self.C < 1:
            raise ValueError(
                "The number of filter channels must be greater than 1.")

        # Calculate the number of partitions for the auralization filter and the
        # FC filter
        self.K_filt = get_num_partitions(self.FL_AUR, self.B)
        self.K_fc = get_num_partitions(self.FL_FC, self.B)

        # Create the filter blocks for the auralization filter and the FC filter
        self.aur_filters_fd = create_filter_blocks(aur_filter_td, self.K_filt, self.B)
        self.fc_filters_fd = create_filter_blocks(fc_filter_td, self.K_fc, self.B)

        # Initialize the frequency-domain delay line (FDL) for the auralization filter and the FC filter
        self.aur_fdl = torch.zeros((self.B + 1, self.K_filt), dtype=torch.complex64)
        self.aur_fdl_cursor = 0

        self.fc_fdl = torch.zeros((self.B + 1, self.K_fc), dtype=torch.complex64)
        self.fc_fdl_cursor = 0

        # Initialize the input buffer
        self.input_buffer_td = torch.zeros(2 * self.B, dtype=self.dtype)

    def parse_input(self, signal: np.ndarray) -> torch.Tensor:
        """
        Parse the input signal and return the FFT transformed signal. The input signal
        is left-shifted by B samples and the new signal is placed at the end of the input
        buffer.

        :param signal: The input signal (shape: (B,))
        :return: The FFT transformed signal (shape: (B + 1))
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
        return input_fd

    def convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform the partitioned convolution
        :param signal: The input signal (shape: (B,))
        :return: The output signal (shape: (C, B + 1))
        """
        # Parse the input signal
        input_fd = self.parse_input(signal)

        # Perform the actual convolution
        output_fd = self.perform_auralization(input_fd)
        self.fdl_cursor = (self.fdl_cursor + 1) % self.K  # Update the index

        # Perform the inverse RFFT to obtain the output signal
        output_td = torch.fft.irfft(output_fd, axis=1)  # shape: (C, 2 * B)

        # Only return the valid samples
        return output_td[:, self.B:].T

    def perform_auralization(self, input_fd: torch.Tensor | np.ndarray) -> torch.Tensor:
        raise NotImplementedError(
            "This method is not implemented in this class.")


# CPU implementation
class PartitionedAuralizationCPU(PartitionedAuralization):

    def __init__(self, aur_filter_td: torch.Tensor, fc_filter_td: torch.Tensor, block_length_samples: int, dtype: np.dtype = torch.float64):
        """
        Initialize the partitioned auralization class
        :param aur_filter_td: The auralization filter in the time domain (shape: (C, FL_AUR))
        :param fc_filter_td: The feedback cancelation filter in the time domain (shape: (L, FL_FC))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        PartitionedAuralization.__init__(
            self, aur_filter_td, fc_filter_td, block_length_samples, dtype)

        self.temp_buffer = np.empty(
            (self.K, self.filters_fd.shape[0], self.filters_fd.shape[1]), dtype=np.complex64)

        # Convert to numpy array for faster computation with Numba
        self.aur_fdl = self.aur_fdl.numpy()
        self.aur_filters_fd = self.aur_filters_fd.numpy()
        self.fc_fdl = self.fc_fdl.numpy()
        self.fc_filters_fd = self.fc_filters_fd.numpy()

    def perform_convolution(self, input_fd: torch.Tensor | np.ndarray) -> torch.Tensor:

        if isinstance(input_fd, np.ndarray):
            input_fd = torch.tensor(input_fd)

        # Store the fd signal in a frequency-domain delay line
        self.aur_fdl[:, self.aur_fdl_cursor] = input_fd

        # Perform the complex multiplication between the fdl and the filter partitions
        output_fd = cpu_aur_multiply(self.aur_filters_fd, self.aur_fdl, self.aur_fdl_cursor,
                                     self.fc_filters_fd, self.fc_fdl, self.fc_fdl_cursor,
                                     self.K, self.temp_buffer)

        # fc_fdl has to contain the sum of all output_fd values
        return torch.from_numpy(output_fd)
