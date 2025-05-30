"""
Partitioned Auralization

This module implements a partitioned algorithm for real-time auralization system.
The algorithm consists of two main components: the auralization filter and the
feedback cancellator.
"""
import numpy as np
import torch

from partitioned_convolution import PartitionedConvolutionCPU, PartitionedConvolutionGPU


# Main class
class PartitionedAuralization:
    """
    Partitioned Auralization class

    This class implements a partitioned algorithm for real-time auralization system. The
    algorithm consists of two main components: the auralization filter and the feedback
    cancellator. The auralization filter is used to simulate the room acoustics, while the
    feedback cancellator is used to cancel the feedback signal.

    Currently, only feedback-cancelation of a single input channel is supported (i.e., SIMO 
    auralization systems). The algorithm can be extended to support multiple input channels
    (MIMO auralization systems) by extending the feedback cancellator to support multiple input 
    channels. For each input channel, a total of C feedback cancellators are required (where C
    is the number of output channels).
    """

    def __init__(self, aur_filter_td: torch.Tensor, fc_filter_td: torch.Tensor,
                 block_length_samples: int, device: str = 'cpu'):
        """
        Initialize the partitioned auralization class
        :param aur_filter_td: The auralization filter in the time domain (shape: (C, FL_AUR))
        :param fc_filter_td: The feedback cancelation filter in the time domain (shape: (C, FL_FC))
        :param block_length_samples: The block length B
        :param device: The device to use ('cpu' or 'gpu')
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
        self.device = device

        fft_size = 2 * self.B

        # Create the partitioned convolution objects
        if self.device == 'cpu':
            self.pc_aur = PartitionedConvolutionCPU(aur_filter_td, block_length_samples,
                                                    fft_size=fft_size, num_input_channels=1)
            self.pc_fc = PartitionedConvolutionCPU(fc_filter_td, block_length_samples,
                                                   fft_size=fft_size, num_input_channels=self.C)
        elif self.device == 'gpu':
            self.pc_aur = PartitionedConvolutionGPU(aur_filter_td, block_length_samples,
                                                    fft_size=fft_size, num_input_channels=1)
            self.pc_fc = PartitionedConvolutionGPU(fc_filter_td, block_length_samples,
                                                   fft_size=fft_size, num_input_channels=self.C)
        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        # Initialize the input buffer
        self.input_buffer_td = torch.zeros(1, fft_size, dtype=torch.float32)
        # Set feedback estimate to zero
        self.feedback_est_td = torch.zeros(1, self.B, dtype=torch.float32)

    def auralize(self, signal_td: np.ndarray) -> np.ndarray:
        """
        Perform the auralization by removing the feedback path from the input and performing
        convolution with the auralization filter. The feedback path is estimated using the
        feedback cancelation filter.
    
        :param signal: The input signal (shape: (1, B) or (B,))
        :return: The auralization output (shape: (C, B))
        """
        if not isinstance(signal_td, torch.Tensor):
            if isinstance(signal_td, np.ndarray):
                signal_td = torch.from_numpy(signal_td.astype(np.float32))
            else:
                raise ValueError("The input signal must be a numpy array or a torch tensor.")

        # Reshape the input signal if necessary
        if signal_td.dim() == 1 and signal_td.shape[0] == self.B:
            signal_td = signal_td.reshape(1, -1)  # shape: (1, B)
        elif signal_td.dim() == 2 and signal_td.shape != (1, self.B):
            raise ValueError("The input signal must have shape (1, B) or (B,).")

        # subtract the feedback signal from the input signal
        signal_td = signal_td - self.feedback_est_td

        # Transform the input signal to the frequency-domain
        signal_fd = self.pc_aur.__parse_input__(signal_td)

        # Perform the actual convolution with the auralization filters
        aur_output_fd = self.pc_aur.__perform_convolution__(signal_fd)

        # Perform the inverse RFFT to obtain the output signal
        aur_output_td = torch.fft.irfft(aur_output_fd, axis=1)[:, -self.B:] # shape: (C, B)

        # Transform the output signal to the frequency-domain for feedback cancelation
        input_fc_td = self.pc_fc.__parse_input__(aur_output_td)

        # Perform the actual convolution with the feedback cancelation filters
        feedback_est_fd = self.pc_fc.__perform_convolution__(input_fc_td)

        # Perform the inverse RFFT to obtain the feedback estimate
        self.feedback_est_td = torch.fft.irfft(feedback_est_fd, axis=1)[:, -self.B:].sum(axis=0)  # shape: (B,)

        if self.device == 'gpu':
            # Move the output spectrum to the CPU
            aur_output_td = aur_output_td.cpu()
            # Move the feedback estimate to the CPU
            self.feedback_est_td = self.feedback_est_td.cpu()

        # Return the auralization output
        return aur_output_td
