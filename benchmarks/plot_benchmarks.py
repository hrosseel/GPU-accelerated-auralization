"""
This script plots the benchmark results for different filter lengths, block sizes, and number of channels for the
partitioned convolution algorithm.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import scienceplots

# plt.style.use(['science', 'ieee'])

FS = 48_000

def plot_benchmark_results(benchmark_data, x_values, x_label="", y_label="", legend_labels=[], figure_filepath="figure.pdf"):
    # Extract the data
    cpu_logs = benchmark_data['cpu_logs']
    gpu_logs = benchmark_data['gpu_logs']

    # Plot the benchmark results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, [np.mean(log) for log in cpu_logs], label=legend_labels[0], marker='o')
    plt.plot(x_values, [np.mean(log) for log in gpu_logs], label=legend_labels[1], marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(figure_filepath)

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
 
    # Plot benchmark results for different filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_conv_filter_len.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['filter_lengths'] / FS,
        x_label='Auralization filter length (s)',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_conv_filter_len.pdf')
    )

    # Plot benchmark results for different block sizes
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_conv_block_size.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['block_sizes'],
        x_label='Block size (samples)',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_conv_block_size.pdf')
    )

    # Plot benchmark results for different number of channels
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_conv_num_channels.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['num_channels'],
        x_label='Number of channels',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_conv_num_channels.pdf')
    )

    # Plot benchmark results for different auralization filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_filter_len_aur.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['aur_filter_lengths'] / FS,
        x_label='Auralization filter length (s)',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_filter_len_aur.pdf')
    )

    # Plot benchmark results for different feedback cancellation filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_filter_len_fc.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['fc_filter_lengths'] / FS,
        x_label='Feedback cancellation filter length (s)',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_filter_len_fc.pdf')
    )

    # Plot benchmark results for different number of channels in auralization
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_num_channels.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['num_channels'],
        x_label='Number of channels',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_num_channels.pdf')
    )

    # Plot benchmark results for different block sizes in auralization
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_block_size.npz'), allow_pickle=True)
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['block_sizes'],
        x_label='Block size (samples)',
        y_label='Time (s)',
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_block_size.pdf')
    )

if __name__ == "__main__":
    main()

