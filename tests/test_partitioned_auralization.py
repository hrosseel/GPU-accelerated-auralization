import pytest
import numpy as np
import torch

from partitioned_auralization import PartitionedAuralization
from lib.acoustic_feedback_simulator import AcousticFeedbackSimulator



def buf_to_blocks(buf: np.ndarray, block_length: int) -> np.ndarray:
    """
    Convert a buffer to blocks
    :param buf: The buffer (shape: (N,))
    :param block_length: The block length
    :param num_blocks: The number of blocks to extract 
    :return: The blocks (shape: (N // block_length, block_length))
    """
    num_blocks = buf.shape[0] // block_length
    blocks = np.zeros((num_blocks, block_length), dtype=buf.dtype)
    for i in range(num_blocks):
        blocks[i, :] = buf[i * block_length:(i + 1) * block_length]
    return blocks

@pytest.fixture(autouse=True)
def setup_method():
    block_length_samples = 1024
    device = 'cpu'
    dtype = torch.float32
    return block_length_samples, device, dtype


def test_single_channel_no_feedback(setup_method):
    block_length_samples, device, dtype = setup_method
    C = 1
    FL_AUR = 1024

    aur_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    fc_filter_td = torch.zeros(C, FL_AUR, dtype=dtype)  # No feedback
    input_block_td = torch.randn(1, block_length_samples, dtype=dtype)

    expected_output = np.convolve(input_block_td.numpy().flatten(), aur_filter_td.numpy().flatten(), mode='full')
    expected_output_blocks = buf_to_blocks(expected_output, block_length_samples)

    pa = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, device)

    output = pa.auralization(input_block_td)
    np.testing.assert_allclose(output.numpy().flatten(), expected_output_blocks[0], rtol=1e-5, atol=1e-5)
    # Check that the output is the same for the next blocks
    for ref_output_block in expected_output_blocks[1:]:
        output = pa.auralization(torch.zeros_like(input_block_td))
        np.testing.assert_allclose(output.numpy().flatten(), ref_output_block, rtol=1e-5, atol=1e-5)


def test_single_channel_feedback(setup_method):
    block_length_samples, device, dtype = setup_method
    C = 1
    FL_AUR = 1024

    aur_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    fc_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    input_block_td = torch.randn(block_length_samples, dtype=dtype)

    # Assuming perfect feedback cancellation
    expected_output = np.convolve(input_block_td, aur_filter_td.numpy().flatten(), mode='full')
    expected_output_blocks = buf_to_blocks(expected_output, block_length_samples)

    # Set up PartitionedAuralization
    pa = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, device)

    # Set up AcousticFeedbackSimulator
    fb_simulator = AcousticFeedbackSimulator(fc_filter_td.numpy(), block_length_samples, expected_output_blocks.shape[0])

    # Auralize the first block
    output = pa.auralization(input_block_td)
    np.testing.assert_allclose(output.numpy().flatten(), expected_output_blocks[0], rtol=1e-3, atol=1e-3)

    # Check that the output is the same for the next blocks (assuming perfect feedback cancellation, with zero input)
    for ref_output_block in expected_output_blocks[1:]:
        # Simulate the feedback path
        input_block = fb_simulator.simulate(output)
        output = pa.auralization(input_block)
        np.testing.assert_allclose(output.numpy().flatten(), ref_output_block, rtol=1e-2, atol=1e-2)


def test_multi_channel_no_feedback(setup_method):
    block_length_samples, device, dtype = setup_method
    C = 24
    FL_AUR = 10_000

    aur_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    fc_filter_td = torch.zeros(C, FL_AUR, dtype=dtype)  # No feedback
    input_block_td = torch.randn(block_length_samples, dtype=dtype)

    expected_output = np.zeros((C, block_length_samples + FL_AUR - 1))
    expected_output_blocks = []
    for i in range(C):
        expected_output[i, :] = np.convolve(input_block_td.numpy(), aur_filter_td[i, :].numpy(), mode='full')
        expected_output_blocks.append(buf_to_blocks(expected_output[i, :], block_length_samples))
    
    expected_output_blocks = np.array(expected_output_blocks).swapaxes(0, 1)

    pa = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, device)

    output = pa.auralization(input_block_td)
    np.testing.assert_allclose(output, expected_output_blocks[0], rtol=1e-3, atol=1e-3)
    # Check that the output is the same for the next blocks
    for ref_output_block in expected_output_blocks[1:]:
        output = pa.auralization(torch.zeros_like(input_block_td))
        np.testing.assert_allclose(output.numpy(), ref_output_block, rtol=1e-2, atol=1e-2)