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

FS = 48_000

# For AES paper:
TEXT_WIDTH = 234.0 / 72.27  # inches
TEXT_HEIGHT = TEXT_WIDTH * (8/10)


def plot_benchmark_results(benchmark_data_conv, benchmark_data_aur, x_values, x_label="", y_label="",
                           latency_budget = [], legend_labels=[], legend_voffset=1):
    # Extract the data
    cpu_logs_conv = benchmark_data_conv['cpu_logs']
    gpu_logs_conv = benchmark_data_conv['gpu_logs']

    cpu_logs_aur = benchmark_data_aur['cpu_logs']
    gpu_logs_aur = benchmark_data_aur['gpu_logs']

    fig, ax = plt.subplots(1, 1, layout='tight', dpi=300)
    # Plot the benchmark results (CPU)
    ax.plot(x_values, np.mean(cpu_logs_conv, axis=1), label=legend_labels[0], linestyle='solid')
    ax.fill_between(x_values, np.min(cpu_logs_conv, axis=1), np.max(cpu_logs_conv, axis=1), alpha=0.2)
    ax.plot(x_values, np.mean(cpu_logs_aur, axis=1), label=legend_labels[1], linestyle='dotted')
    ax.fill_between(x_values, np.min(cpu_logs_aur, axis=1), np.max(cpu_logs_aur, axis=1), alpha=0.2)
    # Plot the benchmark results (GPU)
    ax.plot(x_values, np.mean(gpu_logs_conv, axis=1), label=legend_labels[2], linestyle='solid')
    ax.fill_between(x_values, np.min(gpu_logs_conv, axis=1), np.max(gpu_logs_conv, axis=1), alpha=0.2)
    ax.plot(x_values, np.mean(gpu_logs_aur, axis=1), label=legend_labels[3], linestyle='dotted')
    ax.fill_between(x_values, np.min(gpu_logs_aur, axis=1), np.max(gpu_logs_aur, axis=1), alpha=0.2)
    
    # Plot the latency budget (thick)
    if len(latency_budget) > 0:
        x_values_interp = np.linspace(x_values[0], x_values[-1], 100)
        latency = x_values_interp / FS
        ax.plot(x_values_interp, latency, label='Latency budget', color='grey', linestyle='dashed')
    else:
        latency = [DEFAULT_BLOCK_SIZE / FS] * len(x_values)
        ax.plot(x_values, latency, label='Latency budget', color='grey', linestyle='dashed')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1, legend_voffset))
    ax.set_yscale('log')
    fig.tight_layout()
    fig.set_size_inches(TEXT_WIDTH, TEXT_HEIGHT)
    return fig


def main():
    """ Main function """
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    legend_labels = ['Part. conv. (CPU)', 'Aur. System (CPU)', 'Part. conv. (GPU)', 'Aur. System (GPU)']

    # Plot benchmark results for different filter lengths
    benchmark_data_conv = np.load(os.path.join(workspace_dir, './data/benchmark_conv_filter_len.npz'), allow_pickle=True)
    benchmark_data_aur = np.load(os.path.join(workspace_dir, './data/benchmark_aur_filter_len_aur.npz'), allow_pickle=True)
    
    fig = plot_benchmark_results(
        benchmark_data_conv=benchmark_data_conv,
        benchmark_data_aur=benchmark_data_aur,
        x_values=benchmark_data_conv['filter_lengths'] / FS,
        x_label='Auralization filter length (sec.)',
        y_label='Processing time (sec.)',
        legend_labels=legend_labels,
        legend_voffset=0.885)
    fig.axes[0].set_ylim([1e-4, 1.22e-1])
    fig.savefig(os.path.join(workspace_dir, './figures/benchmark_filter_len.pdf'))

    # Plot benchmark results for different block sizes
    benchmark_data_conv = np.load(os.path.join(workspace_dir, './data/benchmark_conv_block_size.npz'), allow_pickle=True)
    benchmark_data_aur = np.load(os.path.join(workspace_dir, './data/benchmark_aur_block_size.npz'), allow_pickle=True)
    latency_budget = benchmark_data_conv['block_sizes'] / FS

    fig = plot_benchmark_results(
        benchmark_data_conv=benchmark_data_conv,
        benchmark_data_aur=benchmark_data_aur,
        x_values=benchmark_data_conv['block_sizes'],
        x_label='Block size (samples)',
        y_label='Processing time (sec.)',
        latency_budget=latency_budget,
        legend_labels=legend_labels,
        legend_voffset=0.78
    )
    fig.savefig(os.path.join(workspace_dir, './figures/benchmark_block_size.pdf'))

    # Plot benchmark results for different number of channels
    benchmark_data_conv = np.load(os.path.join(workspace_dir, './data/benchmark_conv_num_channels.npz'), allow_pickle=True)
    benchmark_data_aur = np.load(os.path.join(workspace_dir, './data/benchmark_aur_num_channels.npz'), allow_pickle=True)
    fig = plot_benchmark_results(
        benchmark_data_conv=benchmark_data_conv,
        benchmark_data_aur=benchmark_data_aur,
        x_values=benchmark_data_conv['num_channels'],
        x_label="Number of channels",
        y_label='Processing time (sec.)',
        legend_labels=legend_labels,
        legend_voffset=0.77
    )
    fig.savefig(os.path.join(workspace_dir, './figures/benchmark_num_channels.pdf'))

    # # Plot benchmark results for different feedback cancellation filter lengths
    benchmark_data = np.load(os.path.join(workspace_dir, './data/benchmark_aur_filter_len_fc.npz'), allow_pickle=True)
    latency_budget = np.ones(len(benchmark_data['fc_filter_lengths'])) * DEFAULT_BLOCK_SIZE / FS
    
    # Extract the data
    cpu_logs = benchmark_data['cpu_logs']
    gpu_logs = benchmark_data['gpu_logs']

    x_values = benchmark_data['fc_filter_lengths'] / FS

    fig, ax = plt.subplots(1, 1, layout='tight', dpi=300)
    # Plot the benchmark results (CPU)
    ax.plot(x_values, np.mean(cpu_logs, axis=1), label=legend_labels[1], linestyle='dotted', color="#00B945")
    ax.fill_between(x_values, np.min(cpu_logs, axis=1), np.max(cpu_logs, axis=1), alpha=0.2, color="#00B945")
    # Plot the benchmark results (GPU)
    ax.plot(x_values, np.mean(gpu_logs, axis=1), label=legend_labels[3], linestyle='dotted', color="#FF2C00")
    ax.fill_between(x_values, np.min(gpu_logs, axis=1), np.max(gpu_logs, axis=1), alpha=0.2, color="#FF2C00")
    
    # Plot the latency budget
    latency = [DEFAULT_BLOCK_SIZE / FS] * len(x_values)
    ax.plot(x_values, latency, label='Latency budget', color='grey', linestyle='dashed')

    ax.set_xlabel("Feedback cancellation filter length (sec.)")
    ax.set_ylabel("Processing time (sec.)")
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1, 0.8))
    ax.set_yscale('log')
    fig.tight_layout()
    fig.set_size_inches(TEXT_WIDTH, TEXT_HEIGHT)
    fig.savefig(os.path.join(workspace_dir, './figures/benchmark_filter_len_fc.pdf'))


if __name__ == "__main__":
    main()
 