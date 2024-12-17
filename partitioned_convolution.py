import numpy as np


# Create class to perform the convolution
class PartitionedConvolution:

    def __init__(self, filter_td: np.ndarray, block_length_samples: int, dtype: np.dtype = np.float32):
        """
        Initialize the partitioned convolution class
        :param filter_td: The filter in the time domain (shape: (FL, C))
        :param block_length_samples: The block length B
        :param dtype: The data type
        """
        if filter_td.ndim != 2:
            raise ValueError(
                "The filter must be a 2D array with shape (filter_length, num_channels).")

        self.B = block_length_samples
        self.dtype = dtype
        self.FL, self.C = filter_td.shape

        # Validate if FL > B
        if self.FL < self.B:
            raise ValueError(
                "The filter length must be greater than the block length.")
        # Validate the data type
        if self.dtype not in [np.float32, np.float64]:
            raise ValueError("The data type must be np.float32 or np.float64.")
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
        self.filter_parts = np.pad(filter_td, ((0, self.K * self.B - self.FL), (0, 0)),
                                   mode='constant').reshape(self.K, self.B, self.C)

        # Validate the filter length
        if self.filter_parts.shape != (self.K, self.B, self.C):
            raise ValueError(
                "The filter partitions must be equal to the block length")

        # Transform the filter to the frequency domain
        self.filters_fd = self.transform_rfft_filter(self.filter_parts)

        # Initialize the frequency-domain delay line (FDL)
        self.fdl = np.zeros((self.K, self.B + 1), dtype=np.complex64)
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

        # Store the fd signal in a frequency-domain delay line
        self.fdl[self.fdl_cursor, :] = input_fd

        # Perform the complex multiplication between the fdl and the filter partitions
        output_fd = np.zeros((self.B + 1, self.C), dtype=np.complex64)
        for k in range(self.K):
            cursor = (self.fdl_cursor - k) % self.K
            output_fd += (self.fdl[cursor, :, np.newaxis]
                          * self.filters_fd[k, :])

        self.fdl_cursor = (self.fdl_cursor + 1) % self.K  # Update the index

        # Perform the inverse RFFT to obtain the output signal
        output_td = np.fft.irfft(output_fd, axis=0)  # shape: (2 * B,)
        return output_td[self.B:, :]

    def transform_rfft_filter(self, filters_td: np.ndarray) -> np.ndarray:
        # Partition the filter into blocks of length B, and zero-pad another B samples
        filters_padded = np.pad(
            filters_td, ((0, 0), (0, self.B), (0, 0)), mode='constant')  # shape: (K, 2 * B, C)

        # Compute the RFFT of the filters (real-to-complex FFT)
        return np.fft.rfft(filters_padded, axis=1)  # shape: (K, B + 1, C)
