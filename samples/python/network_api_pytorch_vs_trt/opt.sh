#!/bin/bash

# batch sizes and 2 types of infer. backends
bss=("1" "64" "128" "256" "512" "1024")
backs=(0 1)

# save result
output="./data/"
mkdir -p "$output" || { echo "Unable to create the directory"; exit 1; }

# test loop
for back in "${backs[@]}"; do
    for bs in "${bss[@]}"; do
        echo "Testing batch size: $bs"
        python sample.py --use_trt "$back" --bs "$bs" &>> "${output}/$back.log"
    done
done

echo "All tests completed!"
echo "Results saved in: $output"