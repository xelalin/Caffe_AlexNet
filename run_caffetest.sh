#!/bin/bash
source run_fpga_env.sh
./build/tools/caffe test --model=models/bvlc_reference_caffenet/AlexOwt_iter_450000.prototxt.quantized --weights=models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel --iterations=10
