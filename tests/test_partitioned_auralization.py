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

def calc_NMSE(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the NMSE between two signals.
    :param x: The first signal.
    :param y: The second signal.
    :return: The NMSE.
    """
    return np.linalg.norm(x - y, axis=1) ** 2 / np.mean(np.linalg.norm(y, axis=1) ** 2)

@pytest.fixture(autouse=True)
def setup_method():
    block_length_samples = 1024
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
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

    output = pa.auralize(input_block_td)
    np.testing.assert_allclose(output.numpy().flatten(), expected_output_blocks[0], rtol=1e-4, atol=1e-4)
    # Check that the output is the same for the next blocks
    for ref_output_block in expected_output_blocks[1:]:
        output = pa.auralize(torch.zeros_like(input_block_td))
        np.testing.assert_allclose(output.numpy().flatten(), ref_output_block, rtol=1e-4, atol=1e-4)


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
    output = pa.auralize(input_block_td)
    np.testing.assert_allclose(output.numpy().flatten(), expected_output_blocks[0], rtol=1e-3, atol=1e-3)

    # Check that the output is the same for the next blocks (assuming perfect feedback cancellation, with zero input)
    for ref_output_block in expected_output_blocks[1:]:
        # Simulate the feedback path
        input_block = fb_simulator.simulate(output)
        output = pa.auralize(input_block)
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

    output = pa.auralize(input_block_td)
    np.testing.assert_allclose(output, expected_output_blocks[0], rtol=1e-3, atol=1e-3)
    # Check that the output is the same for the next blocks
    for ref_output_block in expected_output_blocks[1:]:
        output = pa.auralize(torch.zeros_like(input_block_td))
        np.testing.assert_allclose(output.numpy(), ref_output_block, rtol=1e-2, atol=1e-2)


def test_multi_channel_feedback(setup_method):
    block_length_samples, device, dtype = setup_method
    C = 24
    FL_AUR = 10_000
    aur_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    fc_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    input_block_td = torch.randn(block_length_samples, dtype=dtype)

    output_length = (1 + (block_length_samples + FL_AUR - 1) // block_length_samples) * block_length_samples
    output_buffer = np.zeros((C, output_length))
    expected_output = np.zeros((C, output_length))

    num_blocks = output_length // block_length_samples
    
    for i in range(C):
        expected_output[i, :(block_length_samples + FL_AUR - 1)] = np.convolve(input_block_td.numpy(), aur_filter_td[i, :].numpy(), mode='full')

    # Set up PartitionedAuralization
    pa = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, device)

    # Set up AcousticFeedbackSimulator
    fb_simulator = AcousticFeedbackSimulator(fc_filter_td.numpy(), block_length_samples, num_blocks)

    # Auralize the first block
    output = pa.auralize(input_block_td)
    output_buffer[:, :block_length_samples] = output.numpy()

    for i in range(1, num_blocks):
        input_block = fb_simulator.simulate(output).sum(axis=0)
        output = pa.auralize(input_block)
        output_buffer[:, i * block_length_samples:(i + 1) * block_length_samples] = output.numpy()

    assert (calc_NMSE(output_buffer, expected_output) < 1e-3).all()