"""
Python class to simulate the acoustic feedback path for auralization systems.
"""
import numpy as np


class AcousticFeedbackSimulator:
    """
    Class to simulate the acoustic feedback path for auralization systems.
    """
    def __init__(self, fb_filter: np.ndarray, block_length: int, num_blocks: int):
        """
        :param fb_filter: Feedback filter to simulate the acoustic feedback path.
        :param block_length: Block length for the simulation.
        :param num_blocks: Number of blocks to simulate.
        """
        self.fb_filter = fb_filter
        self.num_fb_channels, self.fb_length = fb_filter.shape
    
        self.block_length = block_length
        self.fb_buffer = np.zeros((self.num_fb_channels, fb_filter.shape[1] + block_length * num_blocks))
        self.num_blocks = num_blocks
        self.block_cursor = 0

    def simulate(self, input_block: np.ndarray) -> np.ndarray:
        """
        Simulate the acoustic feedback path.
        :param input_block: The input block.
        :return: The output block.
        """

        # Check if block cursor is out of bounds
        if self.block_cursor > self.num_blocks * self.block_length:
            raise ValueError('Maximum number of blocks reached.')

        for i in range(self.num_fb_channels):
            feedback_block = np.convolve(input_block[0, :], self.fb_filter[i, :], mode='full')
            self.fb_buffer[i, self.block_cursor:self.block_cursor + feedback_block.shape[0]] += feedback_block
        output = self.fb_buffer[:, self.block_cursor:self.block_cursor + self.block_length]
        self.block_cursor += self.block_length
        return output
