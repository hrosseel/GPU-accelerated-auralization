#!/bin/bash

# Print "Running benchmarks"
echo "Running benchmarks"

# Activate python
source ~/Documents/hrosseel/python/.venv/bin/activate

# Run the first benchmark using cgexec
cgexec -g cpuset:realtime_app python ~/Documents/hrosseel/python/benchmarks/benchmark_part_conv.py

# sleep for 2 minutes to allow the system to cool down
sleep 120

# Run the second benchmark using cgexec
cgexec -g cpuset:realtime_app python ~/Documents/hrosseel/python/benchmarks/benchmark_part_aur.py

# Print "Benchmarks completed"
echo "Benchmarks completed"
