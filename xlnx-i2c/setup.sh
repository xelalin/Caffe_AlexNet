export XILINX_OPENCL=/opt/ristretto_fpga/xlnx-i2c
export LD_LIBRARY_PATH=$XILINX_OPENCL/runtime/lib/x86_64:$LD_LIBRARY_PATH
export PATH=$XILINX_OPENCL/runtime/bin:$PATH
unset XILINX_SDACCEL
unset XILINX_SDX
unset XCL_EMULATION_MODE
