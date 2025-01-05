from numba import njit, prange, complex64, int64
import numpy as np
import torch
import os

from partitioned_convolution import PartitionedConvolutionCPU, PartitionedConvolutionGPU


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

    def __init__(self, aur_filter_td: torch.Tensor, fc_filter_td: torch.Tensor, block_length_samples: int, device: str = 'cpu',
                 dtype: np.dtype = torch.float64):
        """
        Initialize the partitioned auralization class
        :param aur_filter_td: The auralization filter in the time domain (shape: (C, FL_AUR))
        :param fc_filter_td: The feedback cancelation filter in the time domain (shape: (L, FL_FC))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        if aur_filter_td.ndim != 2 or fc_filter_td.ndim != 2:
            raise ValueError("The input filters must be a 2D array with shape (num_channels, filter_length).")
        if aur_filter_td.shape[0] != fc_filter_td.shape[0]:
            raise ValueError("The number of channels in the auralization filter and "
                             "feedback cancelation filter must be the same.")

        # Set number of channels, filter lengths, and block length
        self.C, self.FL_AUR = aur_filter_td.shape
        _, self.FL_FC = fc_filter_td.shape
        self.B = block_length_samples

        # Create the partitioned convolution objects
        if device == 'cpu':
            self.pc_aur = PartitionedConvolutionCPU(aur_filter_td, block_length_samples, dtype)
            self.pc_fc = PartitionedConvolutionCPU(fc_filter_td, block_length_samples, dtype)
        elif device == 'gpu':
            self.pc_aur = PartitionedConvolutionGPU(aur_filter_td, block_length_samples, dtype)
            self.pc_fc = PartitionedConvolutionGPU(fc_filter_td, block_length_samples, dtype)
        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        # Set feedback estimate to zero
        self.feedback_est_fd = torch.zeros(self.B + 1, dtype=torch.complex64)

    def convolve(self, signal: np.ndarray) -> np.ndarray:
        """
        Convolve the input signal with the auralization filter and cancel the feedback signal using the feedback cancellator
        :param signal: The input signal (shape: (B,))
        :return: The output signal (shape: (C, B + 1))
        """
        # Validate the input signal
        if signal.shape != (self.B,):
            raise ValueError("The input signal must be a 1D array with shape (B,)")

        # Put the incoming signal in the input buffer after sliding the previous signal
        self.input_buffer_td[:self.B] = self.input_buffer_td[self.B:]
        self.input_buffer_td[self.B:] = torch.tensor(signal)

        # Compute the RFFT of the signals (real-to-complex FFT)
        input_fd = torch.fft.rfft(self.input_buffer_td)  # shape: (B + 1)

        # subtract the feedback signal from the input signal
        input_fc_fd = input_fd - self.feedback_est_fd

        # Perform the convolution with the auralization filter
        aur_output_fd = self.pc_aur.__perform_convolution__(input_fc_fd)

        # Update the feedback estimate
        self.feedback_est_fd = self.pc_fc.__perform_convolution__(aur_output_fd).sum(dim=0)





        # Perform the actual convolution
        output_fd = self.__perform_convolution__(input_fd)

        self.fdl_cursor = (self.fdl_cursor + 1) % self.K  # Update the index

        # Perform the inverse RFFT to obtain the output signal
        output_td = torch.fft.irfft(output_fd, axis=1)  # shape: (C, 2 * B)

        # Only return the valid samples
        return output_td[:, self.B:].T
