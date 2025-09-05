# GPU-Accelerated Interactive Auralization of Highly Reverberant Spaces

This repository contains the code for the paper "GPU-Accelerated Interactive Auralization of Highly Reverberant Spaces", which was submitted to the Journal of the Audio Engineering Society. A preprint of this work can be found on [ArXiv](https://arxiv.org/abs/2509.04390).

Authors: Hannes Rosseel and Toon van Waterschoot.

## How to use the code

The code is written in Python 3.13.1. The required packages are listed in requirements.txt. To install the required packages, run the following command:

```bash
pip install -r requirements.txt 
```

The code is organized as follows:
- `benchmarks/`: Contains a Python implementation for performing benchmarks
- `src`: Contains the Python implementation for partitioned auralization and convolution
- `tests`: Contains unit tests

For stable benchmarking, we recommend setting up isolated CPU cores using `cgroups`. An example implementation of using `cgroups` to isolate selected CPU cores can be found below (only works in compatible Linux environments).

```bash
# Limit user and system applications to CPU's 0-3,16-20
systemctl set-property --runtime system.slice AllowedCPUs=0-3,16-20
systemctl set-property --runtime user.slice AllowedCPUs=0-3,16-20

# Create new cgroup with access to other CPU cores
cgcreate -t dspuser:dspuser -a dspuser:dspuser -g cpuset:/realtime_app
cgset -r cpuset.cpus=4-15,20-31 realtime_app
cgset -r cpuset.mems=0 realtime_app

# Set permissions for non-root user execution
chmod o+w  /sys/fs/cgroup/cgroup.procs

# E.g. run an application with the realtime_app cgroup:
# cgexec -g cpuset:/realtime_app <command>
```

The benchmarks can then be run using:
```bash
cgexec -g cpuset:realtime_app python benchmarks/benchmark_part_conv.py
```

## Citation

If you use this code in your research, consider citing the repository and the preprint below:
```
@misc{rosseel2025accelerated,
  title={Accelerated Interactive Auralization of Highly Reverberant Spaces using Graphics Hardware}, 
  author={Hannes Rosseel and Toon van Waterschoot},
  year={2025},
  eprint={2509.04390},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2509.04390}
}
```
