#!/bin/bash
source run_fpga_env.sh

#./build/examples/cpp_classification/classification.bin models/bvlc_reference_caffenet/AlexOwt_iter_450000_dummydata.prototxt.quantized models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg examples/images/cat_gray.jpg examples/images/fish-bike.jpg 16
#./build/tools/run_fpga_flow models/bvlc_reference_caffenet/AlexOwt_iter_450000.prototxt.quantized models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel
#./build/tools/run_fpga_flow models/bvlc_reference_caffenet/AlexOwt_iter_450000_dummydata.prototxt.quantized models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel

#export XBLAS_DAEMON_MODE=1

#./build/tools/run_mp_fpga_flow conv models/bvlc_reference_caffenet/AlexOwt_iter_450000_dummydata.prototxt.quantized models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel
#./build/tools/caffe test --model=models/bvlc_reference_caffenet/AlexOwt_iter_450000.prototxt.quantized --weights=models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel --iterations=25
#./build/examples/cpp_classification/classification.bin models/bvlc_reference_caffenet/AlexOwt_iter_450000_dummydata.prototxt.quantized models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg examples/images/cat_gray.jpg examples/images/fish-bike.jpg 16

./build/examples/cpp_classification/classification.bin models/bvlc_reference_caffenet/AlexOwt_iter_450000_dummydata.prototxt.quantized \
													   models/bvlc_reference_caffenet/AlexOwt_iter_450000.caffemodel \
													   data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt \
./examples/images/cat_gray.jpg ./examples/images/cat_gray.jpg \
./examples/images/cat.jpg	./examples/images/fish-bike.jpg \
1
													   

