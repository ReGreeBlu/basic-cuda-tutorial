#!/usr/bin/env bash
# Run on server w/ CUDA GPU 
set -euo pipefail

# Switch dir to find CUDA source file
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for tool in nvcc ncu; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Cannot find $tool, please have CUDA Toolkit and Nsight Compute installed." >&2
        exit 1
    fi
done

# All the experiments use same vector length
# Change smaller to observe the occupancy of insufficient workload
num_elements=16777216

results_dir=$(mktemp -d "$PWD/occupancy-results.XXXXXX")
echo "Results dir $results_dir"

# Compile once, block size passed through cmd
nvcc -O3 -lineinfo 01-vector-addition.cu -o "$results_dir/vector_add"

for threads in 64 128 256 512; do
    echo
    echo "===== Each thread block has $threads threads, Vector has $num_elements elements ====="

    # See LaunchStats, Occupancy and SpeedOfLight
    # Analyse vectorAdd, print results on terminal, and save reports。
    ncu --kernel-name 'regex:vectorAdd' \
        --launch-count 1 \
        --section LaunchStats \
        --section Occupancy \
        --section SpeedOfLight \
        --page details \
        -o "$results_dir/block_$threads" \
        "$results_dir/vector_add" "$threads" "$num_elements"
done

echo
echo "Finished. Compare each group's Theoretical Occupancy、Achieved Occupancy 和 Duration。"
echo "Reports saved in $results_dir"
