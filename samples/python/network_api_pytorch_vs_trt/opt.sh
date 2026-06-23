#!/bin/bash

if [ "$1" = "0" ]; then
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

elif [ "$1" = "1" ]; then
    bs=1024
    shapes=("1" "28" "128" "256" "512" "1024")
    
    # save result
    output="./data/"
    mkdir -p "$output" || { echo "Unable to create the directory"; exit 1; }

    for shape in "${shapes[@]}"; do
        echo "Testing shape: $shape"
        python sample.py --use_trt 1 --use_exist 0 --bs "$bs" --shape "$shape" &>> "${output}/shapes.log"

    done
else
   echo -e ".sh: nothing\n"
fi
