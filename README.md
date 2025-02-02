For stable benchmarking, we recommend using isolated CPU cores.

```bash
cgexec -g cpuset:realtime_app python benchmarks/benchmark_part_conv.py
```

