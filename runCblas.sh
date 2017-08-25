#!/bin/bash
# set environment LD_LIBRARY_PATH=/opt/ristretto_fpga/xlnx/lib/x86_64:/opt/ristretto_fpga/xlnx/lib
# set environment XILINX_OPENCL=/opt/ristretto_fpga
# set environment XBLAS_RESULT_SHIFT16=1
# set environment CAFFE_XLNX_NUM_IMAGE_CACHE=4

export LD_LIBRARY_PATH=$PWD:/opt/intel/mkl/lib/intel64:/opt/intel/compiler/lib/intel64:$PWD/xlnx-i2c/runtime/lib/x86_64/
export XILINX_OPENCL=$PWD/xlnx-i2c
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
#export SDACCEL_PROFILE_OFF=true

#export XBLAS_SCALE_FIXED_TO_SHORT=1
export XBLAS_USE_CBLAS=1
export XBLAS_RESULT_SHIFT16=1
export XBLAS_NUM_PREP_THREADS=1
export XBLAS_NUM_POST_THREADS=1
#export XBLAS_USE_BATCHED_CPOST=1
#export XBLAS_NUM_KERNELS=1

export CAFFE_XLNX_USE_DEFAULT_SGEMM=1
#export CAFFE_XLNX_NUM_THREADS=4
#export CAFFE_XLNX_SKIP_BIAS=1
#export CAFFE_XLNX_EXCLUDE_FC_TIME=1
export CAFFE_XLNX_NUM_IMAGE_CACHE=64
export CAFFE_XLNX_DO_GEMM_PREP_B=1
export CAFFE_XLNX_HACK_QUANTSCALE=1 
#export CAFFE_XLNX_PIPELINE_GEMM=1
#export CAFFE_XLNX_IMAGE_BATCH_SIZE=8 # enables batching if BATCH_SIZE > 1

echo " "
echo "=============== XBLAS ============================="

#./build/examples/cpp_classification/classification.bin models/bvlc_reference_caffenet/deploy_nogroup.prototxt.quantized models/bvlc_reference_caffenet/bvlc_reference_caffenet_nogroup.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg examples/images/cat_gray.jpg examples/images/fish-bike.jpg examples/images/cat_gray.jpg 16
#./build/examples/cpp_classification/classification.bin models/bvlc_reference_caffenet/deploy_nogroup.prototxt.quantized models/bvlc_reference_caffenet/bill_nogroup.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg examples/images/cat_gray.jpg examples/images/fish-bike.jpg examples/images/cat_gray.jpg 16

./build/examples/cpp_classification/classification.bin ./AlexNet/Alex_quantize16bit_deploy.prototxt ./AlexNet/Alex_Quantize16bit_iter_90000.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg examples/images/cat_gray.jpg examples/images/fish-bike.jpg examples/images/cat_gray.jpg 16
./build/examples/cpp_classification/classification.bin ./AlexNet/Alex_quantize16bit_deploy.prototxt ./AlexNet/Alex_Quantize16bit_iter_90000_nogroup.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg examples/images/cat_gray.jpg examples/images/fish-bike.jpg examples/images/cat_gray.jpg 16
