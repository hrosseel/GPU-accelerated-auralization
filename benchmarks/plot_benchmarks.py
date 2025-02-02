"""
This script plots the benchmark results for different filter lengths, block sizes, and number of channels for the
partitioned convolution algorithm.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import scienceplots


from config import DEFAULT_BLOCK_SIZE

plt.style.use(['science', 'grid', 'ieee', 'std-colors'])

plt.rc('text', usetex=True)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

text_width = 3.48761  # inches
text_height = text_width * (7/10)

FS = 48_000

def plot_benchmark_results(benchmark_data, x_values, x_label="", y_label="", latency_budget = [], legend_labels=[], figure_filepath="figure.pdf"):
    # Extract the data
    cpu_logs = benchmark_data['cpu_logs']
    gpu_logs = benchmark_data['gpu_logs']

    # Plot the benchmark results
    fig, ax = plt.subplots(1, 1, layout='tight')
    ax.plot(x_values, np.mean(cpu_logs, axis=1), label=legend_labels[0])
    ax.plot(x_values, np.mean(gpu_logs, axis=1), label=legend_labels[1], linestyle='--')
    # Shade the region between the minimum and maximum values
    ax.fill_between(x_values, np.min(cpu_logs, axis=1), np.max(cpu_logs, axis=1), alpha=0.3)
    ax.fill_between(x_values, np.min(gpu_logs, axis=1), np.max(gpu_logs, axis=1), alpha=0.3)

    # Plot the latency budget
    if len(latency_budget) > 0:
        ax.plot(x_values, latency_budget, label='Latency budget', linestyle='dashdot', color='black')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=7, loc='best')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.set_size_inches(text_width, text_height)
    fig.savefig(figure_filepath)

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Plot benchmark results for different filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_conv_filter_len.npz'), allow_pickle=True)
    latency_budget = np.ones(len(benchmark_data['filter_lengths'])) * DEFAULT_BLOCK_SIZE / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['filter_lengths'] / FS,
        x_label='Auralization filter length (s)',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_conv_filter_len.pdf')
    )

    # Plot benchmark results for different block sizes
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_conv_block_size.npz'), allow_pickle=True)
    latency_budget = benchmark_data['block_sizes'] / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['block_sizes'],
        x_label='Block size (samples)',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_conv_block_size.pdf')
    )

    # Plot benchmark results for different number of channels
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_conv_num_channels.npz'), allow_pickle=True)
    latency_budget = np.ones(len(benchmark_data['num_channels'])) * DEFAULT_BLOCK_SIZE / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['num_channels'],
        x_label='Number of channels',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_conv_num_channels.pdf')
    )

    # Plot benchmark results for different auralization filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_filter_len_aur.npz'), allow_pickle=True)
    latency_budget = np.ones(len(benchmark_data['aur_filter_lengths'])) * DEFAULT_BLOCK_SIZE / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['aur_filter_lengths'] / FS,
        x_label='Auralization filter length (s)',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_filter_len_aur.pdf')
    )

    # Plot benchmark results for different feedback cancellation filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_filter_len_fc.npz'), allow_pickle=True)
    latency_budget = np.ones(len(benchmark_data['fc_filter_lengths'])) * DEFAULT_BLOCK_SIZE / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['fc_filter_lengths'] / FS,
        x_label='Feedback cancellation filter length (s)',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_filter_len_fc.pdf')
    )

    # Plot benchmark results for different number of channels in auralization
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_num_channels.npz'), allow_pickle=True)
    latency_budget = np.ones(len(benchmark_data['num_channels'])) * DEFAULT_BLOCK_SIZE / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['num_channels'],
        x_label='Number of channels',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_num_channels.pdf')
    )

    # Plot benchmark results for different block sizes in auralization
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_block_size.npz'), allow_pickle=True)
    latency_budget = benchmark_data['block_sizes'] / FS
    plot_benchmark_results(
        benchmark_data=benchmark_data,
        x_values=benchmark_data['block_sizes'],
        x_label='Block size (samples)',
        y_label='Time (s)',
        latency_budget=latency_budget,
        legend_labels=['CPU', 'GPU'],
        figure_filepath=os.path.join(workspace_dir, './figures/benchmark_aur_block_size.pdf')
    )

if __name__ == "__main__":
    main()

