#!/bin/bash

# bss=("1")
bss=("1" "64" "128" "256" "512" "1024")
backs=(0 1)
output="./data/"
mkdir -p "$output" || { echo "Unable to create the directory"; exit 1; }

for back in "${backs[@]}"; do
    for bs in "${bss[@]}"; do
        echo "Testing batch size: $bs"
        python sample.py --inf_back "$back" --bs "$bs" &>> "${output}/$back.log"
    done
done

echo "All tests completed!"