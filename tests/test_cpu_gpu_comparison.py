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
    torch.manual_seed(0)
    block_length_samples = 64
    dtype = torch.float32
    return block_length_samples, dtype


def test_single_channel_no_feedback(setup_method):
    block_length_samples, dtype = setup_method
    C = 1
    FL_AUR = 4096

    aur_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    fc_filter_td = torch.zeros(C, FL_AUR, dtype=dtype)  # No feedback
    input_block_td = torch.randn(1, block_length_samples, dtype=dtype)

    num_blocks = (FL_AUR + block_length_samples - 1) // block_length_samples + 1

    pa_cpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'cpu')
    pa_gpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'gpu')

    output_cpu = pa_cpu.auralize(input_block_td)
    output_gpu = pa_gpu.auralize(input_block_td)

    assert np.max(np.abs(output_cpu.numpy() - output_gpu.numpy())) < 1e-5
    
    # Check that the output is the same for the next blocks
    for _ in range(num_blocks):
        output_cpu = pa_cpu.auralize(torch.zeros_like(input_block_td))
        output_gpu = pa_gpu.auralize(torch.zeros_like(input_block_td))

        # Max difference should be less than 10 * 1e-6 (considering floating point errors)
        assert np.max(np.abs(output_cpu.numpy() - output_gpu.numpy())) < 1e-5


def test_single_channel_feedback(setup_method):
    block_length_samples, dtype = setup_method
    C = 1
    FL_AUR = 4096
    FL_FC = 1024

    aur_filter_td = torch.rand(C, FL_AUR, dtype=dtype)
    aur_filter_td /= torch.norm(aur_filter_td, dim=1)  # Normalize to avoid gain greater than 1
    fc_filter_td = torch.zeros(C, FL_FC, dtype=dtype)  # Feedback filter, modeled as sinc functions

    for i in range(C):
        sample_delay = torch.randint(low=0, high=FL_FC // 4, size=(1,)).item()
        t = torch.arange(FL_FC, dtype=dtype)
        fc_filter_td[i, :] = 0.8 * torch.sinc(0.1 * (t - sample_delay))
        
    input_block_td = torch.rand(block_length_samples, dtype=dtype)

    num_blocks = (FL_AUR + block_length_samples - 1) // block_length_samples + 1

    # Set up PartitionedAuralization
    pa_cpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'cpu')
    pa_gpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'gpu')

    # Set up AcousticFeedbackSimulator
    fb_simulator_cpu = AcousticFeedbackSimulator(fc_filter_td.numpy(), block_length_samples, num_blocks)
    fb_simulator_gpu = AcousticFeedbackSimulator(fc_filter_td.numpy(), block_length_samples, num_blocks)

    # Auralize the first block
    output_cpu = pa_cpu.auralize(input_block_td)
    output_gpu = pa_gpu.auralize(input_block_td)

    for _ in range(num_blocks):
        # Simulate the feedback path
        input_buf_cpu = fb_simulator_cpu.simulate(output_cpu)
        input_buf_gpu = fb_simulator_gpu.simulate(output_gpu)

        output_cpu = pa_cpu.auralize(input_buf_cpu)
        output_gpu = pa_gpu.auralize(input_buf_gpu)

        assert np.max(np.abs(output_cpu.numpy() - output_gpu.numpy())) < 1e-5


def test_multi_channel_no_feedback(setup_method):
    block_length_samples, dtype = setup_method
    C = 24
    FL_AUR = 4096

    aur_filter_td = torch.randn(C, FL_AUR, dtype=dtype)
    fc_filter_td = torch.zeros(C, FL_AUR, dtype=dtype)  # No feedback
    input_block_td = torch.randn(block_length_samples, dtype=dtype)
    
    num_blocks = (FL_AUR + block_length_samples - 1) // block_length_samples + 1

    pa_cpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'cpu')
    pa_gpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'gpu')

    output_cpu = pa_cpu.auralize(input_block_td)
    output_gpu = pa_gpu.auralize(input_block_td)

    assert np.max(np.abs(output_cpu.numpy() - output_gpu.numpy())) < 1e-5
    
    # Check that the output is the same for the next blocks
    for _ in range(num_blocks):
        output_cpu = pa_cpu.auralize(torch.zeros_like(input_block_td))
        output_gpu = pa_gpu.auralize(torch.zeros_like(input_block_td))

        # Max difference should be less than 10 * 1e-6 (considering floating point errors)
        assert np.max(np.abs(output_cpu.numpy() - output_gpu.numpy())) < 1e-5


def test_multi_channel_feedback(setup_method):
    block_length_samples, dtype = setup_method
    C = 24
    FL_AUR = 4096
    FL_FC = 1024

    aur_filter_td = torch.rand(C, FL_AUR, dtype=dtype)
    aur_filter_td /= torch.norm(aur_filter_td, dim=1)[:, torch.newaxis]  # Normalize to avoid gain greater than 1
    fc_filter_td = torch.zeros(C, FL_FC, dtype=dtype)  # Feedback filter, modeled as sinc functions

    for i in range(C):
        sample_delay = torch.randint(low=0, high=FL_FC // 4, size=(1,)).item()
        t = torch.arange(FL_FC, dtype=dtype)
        fc_filter_td[i, :] = 0.8 * torch.sinc(0.1 * (t - sample_delay))
        
    input_block_td = torch.rand(block_length_samples, dtype=dtype)

    num_blocks = (FL_AUR + block_length_samples - 1) // block_length_samples + 1

    # Set up PartitionedAuralization
    pa_cpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'cpu')
    pa_gpu = PartitionedAuralization(aur_filter_td, fc_filter_td, block_length_samples, 'gpu')

    # Set up AcousticFeedbackSimulator
    fb_simulator_cpu = AcousticFeedbackSimulator(fc_filter_td.numpy(), block_length_samples, num_blocks)
    fb_simulator_gpu = AcousticFeedbackSimulator(fc_filter_td.numpy(), block_length_samples, num_blocks)

    # Auralize the first block
    output_cpu = pa_cpu.auralize(input_block_td)
    output_gpu = pa_gpu.auralize(input_block_td)

    for _ in range(num_blocks):
        # Simulate the feedback path
        input_buf_cpu = fb_simulator_cpu.simulate(output_cpu)
        input_buf_gpu = fb_simulator_gpu.simulate(output_gpu)

        output_cpu = pa_cpu.auralize(input_buf_cpu.sum(axis=0))
        output_gpu = pa_gpu.auralize(input_buf_gpu.sum(axis=0))

        assert np.max(np.abs(output_cpu.numpy() - output_gpu.numpy())) < 3e-5
