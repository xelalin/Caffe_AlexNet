#!/bin/bash
# set environment LD_LIBRARY_PATH=/opt/ristretto_fpga/xlnx/lib/x86_64:/opt/ristretto_fpga/xlnx/lib
# set environment XILINX_OPENCL=/opt/ristretto_fpga
# set environment XBLAS_RESULT_SHIFT16=1
# set environment CAFFE_XLNX_NUM_IMAGE_CACHE=4

export LD_LIBRARY_PATH=/opt/OpenBLAS:/opt/zmq/libs:/opt/intel/mkl/lib/intel64:/opt/intel/compiler/lib/intel64:$PWD/xlnx-i2c/runtime/lib/x86_64/:$PWD
export XILINX_OPENCL=$PWD/xlnx-i2c
#export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/opt/intel/compiler/lib/intel64:$PWD/xlnx-memdbg/runtime/lib/x86_64/:$PWD
#export XILINX_OPENCL=$PWD/xlnx-memdbg
#export OMP_NUM_THREADS=8
#export OPENBLAS_NUM_THREADS=8
#export SDACCEL_PROFILE_OFF=true

#export XBLAS_SCALE_FIXED_TO_SHORT=1
#export XBLAS_FORCE_RUN_ON_KERNEL=0
#export XBLAS_USE_CBLAS=1
export XBLAS_USE_BATCHED_CPOST=1
export XBLAS_RESULT_SHIFT16=1
export XBLAS_NUM_PREP_THREADS=1
export XBLAS_NUM_POST_THREADS=1
#export XBLAS_USE_BUFFERED_A_LOADER=1
export XBLAS_DUMP_JZEJDA_PROFILE_DATA=1
export XBLAS_NUM_KERNELS=2
export XBLAS_USE_IM2COL_KERNEL=1
export XBLAS_IM2COL_CPOST_SPLIT_C=1
export XBLAS_DAEMON_MODE=1

#export CAFFE_XLNX_KERNEL_BASE_NAME="kernelSgemm_0"
#export CAFFE_XLNX_USE_DEFAULT_SGEMM=1
#export CAFFE_XLNX_NUM_THREADS=4
#export CAFFE_XLNX_EXCLUDE_FC_TIME=1

#export CAFFE_XLNX_IMAGE_BATCH_SIZE=4 # enables batching if BATCH_SIZE > 1
#export CAFFE_XLNX_DO_GEMM_PREP_B=1
export CAFFE_XLNX_PIPELINE_GEMM=1
export CAFFE_XLNX_HACK_QUANTSCALE=1 
export CAFFE_XLNX_SKIP_BIAS=1
export CAFFE_XLNX_NUM_IMAGE_CACHE=64
export CAFFE_XLNX_DO_FPGA_IM2COL=1
export CAFFE_XLNX_FPGA_IM2COL_BATCH_SIZE=8 # enables im2col batching if BATCH_SIZE > 1
export CAFFE_XLNX_ENABLE_CONVPLUS=1
export CAFFE_XLNX_KEEP_CONV_ON_DDR=1

export CAFFE_XLNX_USE_CBLAS_IF_NOT_CONV=1 

export XBLAS_EMIT_PROFILING_INFO=1

#export XBLAS_USE_CBLAS=1
#export CAFFE_XLNX_USE_DEFAULT_SGEMM=1
#export XBLAS_SKIP_POOL5_CPOST=1
export SDACCEL_INI_PATH=$PWD
