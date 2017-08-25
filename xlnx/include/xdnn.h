// Copyright (c) 2017
// Xilinx, Inc.
// All rights reserved.
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief Xilinx DNN library for FPGA acceleration
 *
 *  @author Aaron Ng (aaronn@xilinx.com)
 */

#ifndef XDNN_H
#define XDNN_H

class XPipelinePacket;
enum XDNNTensorShapeType { XDNN_TENSOR_1D, XDNN_TENSOR_NCHW };
class XDNNDataDescriptor {
public:
  XDNNDataDescriptor();
  XDNNDataDescriptor(void *data, 
    int dataTypeSize,
    XDNNTensorShapeType shapeType,
    int n, int c, int h, int w, int baseAddr);
  int getSize();
  virtual int execute(XPipelinePacket *packet) { assert(0); return -1; }

  void *_data;
  XDNNTensorShapeType _shapeType;
  int _shape[4];
  int _dataTypeSize;
  int _baseAddr;
};

class XDNNInputDescriptor : public XDNNDataDescriptor {
public:
  XDNNInputDescriptor()
  : XDNNDataDescriptor()
  {}
  XDNNInputDescriptor(void *data, 
    int dataTypeSize,
    XDNNTensorShapeType shapeType,
    int n, int c, int h, int w, int baseAddr) 
  : XDNNDataDescriptor(data, dataTypeSize, shapeType, n, c, h, w, baseAddr)
  {}
  virtual int execute(XPipelinePacket *packet);
};

class XDNNOutputDescriptor : public XDNNDataDescriptor {
public:
  XDNNOutputDescriptor()
  : XDNNDataDescriptor()
  {}
  XDNNOutputDescriptor(void *data, 
    int dataTypeSize,
    XDNNTensorShapeType shapeType,
    int n, int c, int h, int w, int baseAddr) 
  : XDNNDataDescriptor(data, dataTypeSize, shapeType, n, c, h, w, baseAddr)
  {}
  virtual int execute(XPipelinePacket *packet);
};

enum XDNNOperation { XDNN_NOOP, XDNN_CONV, XDNN_MAXPOOL };
class XDNNDescriptor {
public:
  // default (user manually fills in)
  XDNNDescriptor();

  XDNNDescriptor(XDNNOperation op, 
    int opsize, int stride, 
    int inBaseAddr, int inImgSize, int inImgChannels,
    int outBaseAddr, int outImgSize, int outImgChannels);

  virtual int execute(XPipelinePacket *packet) { assert(0); return -1; }

  XDNNOperation _op;

  // common args
  int _opSize;
  int _opStride; 
  int _inBaseAddr; 
  int _inImgSize; 
  int _inImgChannels;
  int _outBaseAddr; 
  int _outImgSize; 
  int _outImgChannels; 
};

class XDNNConvolutionDescriptor : public XDNNDescriptor {
public: 
  XDNNConvolutionDescriptor(void *A, int kernSize, int kernStride, 
    int rightShift, bool relu,
    int inBaseAddr, int inImgSize, int inImgChannels,
    int outBaseAddr, int outImgSize, int outImgChannels);
  virtual int execute(XPipelinePacket *packet);

  void *_A;
  int _rightShift; 
  bool _relu;
};
class XDNNMaxpoolDescriptor : public XDNNDescriptor {
public: 
  XDNNMaxpoolDescriptor(int winSize, int winStride, 
    int inBaseAddr, int inImgSize, int inImgChannels,
    int outBaseAddr, int outImgSize, int outImgChannels);
  virtual int execute(XPipelinePacket *packet);
};

class XBLASHandle;
class XBLASConfig;
int fpgaXdnn(XBLASHandle &handle, 
  XDNNInputDescriptor input,
  std::vector<std::shared_ptr<XDNNDescriptor> > descs,
  XDNNOutputDescriptor output,
  XBLASConfig *cfg=NULL);

void XDNNMatToDDR(short *src, int rows, int cols, 
  int kern, int inChans, int outChans,
  short *&out, int &outRows, int &outCols);
void XDNNGetMatToDDRSize(int kern, int inChans, int outChans, 
  int &outRows, int &outCols);

template<typename T>
void XDNNdbgDumpToFile(const T *data, int size, std::string fname);
template<typename T>
void XDNNdbgReadFromFile(T *data, int size, std::string fname);

#endif // XDNN_H

