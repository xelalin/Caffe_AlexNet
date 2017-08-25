// Copyright (c) 2016
// Xilinx, Inc.
// All rights reserved.
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief Xilinx BLAS library for FPGA acceleration
 *
 *  @author Aaron Ng (aaronn@xilinx.com)
 */

/*************************************************************** 
  Example SGEMM usage:

   XBLASHandle* xHandlePtr;
   xblasCreate(xHandlePtr, "kernelSgemm.xclbin", "kernelSgemm");
   XBLASHandle &xHandle = *xHandlePtr;

   size_t aSize = sizeof(short) * numElementsA;
   size_t bSize = sizeof(short) * numElementsB;
   size_t cSize = sizeof(short) * numElementsC;

   XMemPtr *aPtr, *bPtr, *cPtr;
   xMalloc(xHandle, aPtr, aSize);
   xMalloc(xHandle, bPtr, bSize);
   xMalloc(xHandle, cPtr, cSize);

   xMemcpy(xHandle, matrixA, aPtr, aSize);
   xMemcpy(xHandle, matrixB, bPtr, bSize);
   xMemcpy(xHandle, matrixC, cPtr, cSize);

   xblasSgemm<short>(xHandle, 
     CblasRowMajor, CblasNoTrans, CblasNoTrans,
     M, N, K,
     alpha, 
     aPtr, lda,
     bPtr, ldb,
     beta,
     cPtr, ldc);

   xMemcpy(xHandle, cPtr, matrixC, cSize);

   xFree(xHandle, aPtr);
   xFree(xHandle, bPtr);
   xFree(xHandle, cPtr);

   xblasDestroy(xHandle);

 ***************************************************************/

#ifndef XBLAS_H
#define XBLAS_H

#include <chrono>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <CL/opencl.h>
#include <boost/thread.hpp>
#include "xdnn.h"

class XMemPtr;
class XDeviceMemPOD;
template <typename HostT, typename T>
class XHostMatrix;
class XHostMatrixBase;

enum XBLASKernelType { CLASSIC_KERNEL, RTLXDNN_KERNEL };
class XBLASKernelConfig {
public:
  XBLASKernelConfig();

  int m_minM;
  int m_minK;
  //int m_minKa;
  //int m_minKb;
  int m_minN;

  int m_numKernels;
  int m_forceRunOnKernelIdx;
  XBLASKernelType m_kernelType;

  int m_bSplit;
  bool m_useRtlFlow;
  bool m_useBufferedALoader;
  bool m_useDdrKargs;
  bool m_hasIm2Col;
  bool m_hasMaxPool; // only used by RTL kernel
  bool m_hasRelu;    // only used by RTL kernel

  int m_numCoresX;
  int m_numCoresY;
  int m_numCoreColumns;
  int m_numCoreRows;

  // DDR mappings for each kernel
  std::vector<int> m_ddrBankKA; 
  std::vector<int> m_ddrBankA;
  std::vector<std::vector<int> > m_ddrBankB;
  std::vector<int> m_ddrBankC;

  bool m_resultShift16;

  int getDDRMap(int i); 

  void print();

private:
  std::vector<int> m_ddrBankMap;
};

class XBLASIm2ColParams {
public:
  XBLASIm2ColParams();
  bool operator==(const XBLASIm2ColParams &other) const;
  bool operator!=(const XBLASIm2ColParams &other) const;

  int m_doIm2Col;
  int m_batchSize; // this controls im2col-only batching
  int m_imgChannels;
  int m_imgHeight;
  int m_imgWidth;
  int m_imgWidthPadded;
  int m_outWidth;
  int m_outWidthPadded;
  //int m_numKernels;
  
  int m_doMaxPool;
  bool m_doRelu;

  bool m_cPostSplitC;

  // these are "use-once" keys for passing 
  // cl_mem between 2 layers
  // if 'read' key is passed, use B from use-once cache; 
  //   (skip B_prep and delete cl_mem after use)
  // if 'write' key is passed, save C to use-once cache
  //   (and skip C_post)
  std::string m_ddrCacheReadKey;
  std::string m_ddrCacheWriteKey;
  const void *m_ddrCacheReadKeyPtr;
  const void *m_ddrCacheWriteKeyPtr;

  void print();
};

class XBLASConvParams {
public:
  XBLASConvParams();
  int m_M;
  int m_N;
  int m_K;
  unsigned short m_scaleFPGAC;
  unsigned int m_Aoffset; // offset in shorts, aligned
  XBLASIm2ColParams m_i2cParams; // note: last layer's i2cp params 
                                 // will be used for CprepZero and Cpost
};

class XBLASFlowParams {
public:
  XBLASFlowParams();
  XBLASConvParams& getLastConvParam();
  XBLASIm2ColParams& getLastI2cParam();

  std::vector<XBLASConvParams> m_convParams; // for each layer
  int m_BCsize; // size in shorts, user must align to 4096.
                // clCreateBuffer must multiply by sizeof(short) 
  void print();
};

class XBLASConfig {
public:
  XBLASConfig();
  void print();
  bool doAllConvFlow() 
    { return !m_flowParams.m_convParams.empty(); }

  bool m_useCblas;
  bool m_async; // if async, user must call waitForResults()
  bool m_useBatchedCpost;
  float m_scaleA;
  float m_scaleB;
  float m_scaleC;
  unsigned short m_scaleFPGAC;
  bool m_cacheA;
  bool m_cacheB;
  bool m_cacheC;
  bool m_doPrepA;
  bool m_doPrepB;
  int m_batchSize; // this controls non-im2col batching
  int m_batchOutputOffset;
  std::string m_taskName;
  bool m_dumpJzejdaProfileData;
  bool m_dumpGemmC;
  int m_streamId;
  XBLASIm2ColParams m_im2colParams;
  XBLASFlowParams m_flowParams;
};

template <typename T>
class CLObjs {
public:
  CLObjs() : m_objs(NULL), m_numObjs(0) { }
  CLObjs(const CLObjs &obj);
  CLObjs& operator= (const CLObjs &obj);

  ~CLObjs() {
    clear();
  }
  void clear();
  void add(T &obj, std::string label="")
  {
    if (m_objs == NULL)
    {
      m_objs = new T[10];
      m_labels.resize(10);
    }

    assert(m_numObjs < 10);
    m_objs[m_numObjs] = obj;
    m_labels[m_numObjs] = label;
    m_numObjs++;
  }
  void extend(const CLObjs &obj)
  {
    for (int i=0; i < obj.m_numObjs; i++)
      add(obj.m_objs[i], obj.m_labels[i]);
  }

  T *m_objs;
  int m_numObjs;
  std::vector<std::string> m_labels;
};

class XTimer
{
  public:
    XTimer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
      return std::chrono::duration_cast<second_>
        (clock_::now() - beg_).count(); }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

class XPipelinePacket {
public:
  XPipelinePacket();

  void printProfilingInfo();
  void emitProfilingInfo();
  void print();
  void saveToJournal();
  void cleanup();

  int id; 
  XTimer startTime;
  class XBLASHandle *xHandle;
  int transA;
  int transB;
  int M;
  int N;
  int K;
  XMemPtr *A;
  XMemPtr *B;
  XMemPtr *C;
  int aOffset;
  int bOffset;
  int cOffset;
  int la;
  int lb;
  int lc;
  int alpha;
  int beta;
  XBLASConfig cfg;
  XDNNInputDescriptor xdnnInput;
  std::vector<std::shared_ptr<XDNNDescriptor> > xdnnDescriptors;
  XDNNOutputDescriptor xdnnOutput;
  ///// kernel scheduling
  int kernelIdx;
  ///// write stage
  void *kaArgsPtr;
  void *kaArgsAlignedBuf;
  cl_mem kaMemPtr;
  XDeviceMemPOD *aDevMemPOD;
  XDeviceMemPOD *bDevMemPOD;
  XDeviceMemPOD *cDevMemPOD;
  ///// exec stage
  CLObjs<cl_mem> memToWrite;
  CLObjs<cl_event> writeDependencies; // start writes only after these events
  CLObjs<cl_event> readDependencies; // start reads only after these events
  CLObjs<cl_event> execDependencies; // start execs only after these events
  CLObjs<cl_event> writeEvents;
  CLObjs<cl_event> execEvents;
  CLObjs<cl_event> readEvents;

  CLObjs<cl_event> eventsToRelease; // events to release after done with packet
};

class XBLASHandle {
  static XBLASHandle* m_impl; // singleton

public:
  static XBLASHandle* get();

  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_device_id device_id;
  std::vector<cl_kernel> kernels;

  bool m_isInitialized;
  XBLASKernelConfig m_kConfig;
  XBLASConfig m_config;

  // m_host2XMemMap helpers
  void setHost2XMem(const void *src, XMemPtr* dest);
  XMemPtr* getHost2XMem(const void *src);
  void deleteHost2XMem(const void *src);
  void destroyDeviceMemPODs();

  // this is populated automatically; summed from all calls
  void printTaskTimes();
  void addTaskTime(std::string task, double time, 
    XBLASConfig *cfg=NULL);
  std::map<std::string, double> m_taskTimes;

  // thread management
  template <typename HostT, typename T>
  void setupThreads();
  void resetPipelineStartTime();
  void waitForResults(int streamId=0, bool recordPipelineTimes=false);
  int queryStream(int streamId=0);

  // multi-handle management
  void addRefCount();
  int delRefCount();
  void release(bool printTaskTimes=false);

private:
  XBLASHandle(); // handled by get()
  ~XBLASHandle(); // handled by release()

  // store a mapping of all xMemcpy (src, dest) ptrs
  // this is so we can map src ptrs to XMemPtrs.
  std::map<const void*, XMemPtr*> m_host2XMemMap;

  // make one command queue for each thread
  std::map<unsigned long, cl_command_queue> m_commands; 

  // thread stuff
  void destroyThreads();
  bool m_threadsAlive;
  boost::mutex m_timer_mtx;
  boost::mutex m_mtx;
  int m_refCnt;
};

// to capture what matrix we went to the device from XMemPtr.
// If any param of the matrix changes, we have to reformat/re-send to device.
class XMatrixSignature {
public:
  XMatrixSignature();
  ~XMatrixSignature() {}
  bool operator==(const XMatrixSignature &other) const;
  bool operator!=(const XMatrixSignature &other) const;

  int m_offset;
  int m_rows;
  int m_minRows;
  int m_cols;
  int m_minCols;
  int m_ld;
  XBLASIm2ColParams m_im2colParams;
  double m_scale;
  int m_trans;
};

// Class to manage host<->device memory alloc and transfer
// Note: 
// - XMemPtr maps 1-to-1 to host ptr
// - in multi-kernel mode, XMemPtr maps 1-to-many to XDeviceMemPOD objects
//     
class XMemPtr {
public:
  XMemPtr(size_t sz, int numKernels, bool isMappedtoHost=false);
  ~XMemPtr();
  void copyInto(void *data, size_t size);
  void copyOut(void *data, size_t size);
  void xmemset(int alpha, size_t size);

  XDeviceMemPOD* getDeviceMemPOD(void *, int kernelIdx);
  XDeviceMemPOD* makeNewDeviceMemPOD(void *, XMatrixSignature &sig, int kernelIdx, bool cache=true);
  void deleteDeviceMemPODs();

  // populated by user
  size_t m_size;
  size_t m_dataSize;
  void *m_data; // copy of original data, can be replaced after kernel runs
  const void *m_srcPtr; // pointer to original data, in case we need lookup
  bool m_isMappedToHost; // if mapped to host, all writes go 
                         // directly back to m_srcPtr and 
                         // copyOut() is a noop

private:
  // auto-populated when data is written from host to device
  // this holds the data objects prepped for the FPGA
  // m_hostDeviceMemMap helpers
  // Note: vector because each kernel has its own hostDeviceMemMap
  //       (m_hostDeviceMemMap.size() == num kernels)
  std::vector<std::unordered_map<void*, XDeviceMemPOD*> > m_hostDeviceMemMap;

  XMemPtr();
};

// stores temp objs to manage device buffers, handles and writes
class XDeviceMemPOD {
public:
  XDeviceMemPOD();
  ~XDeviceMemPOD();
  void reset();

  // auto-populated when data written to device
  XMatrixSignature m_matrixSig;
  XHostMatrixBase *m_hostObj; // unformatted matrix
  std::vector<XHostMatrixBase*> m_deviceObjs; // formatted matrix
  std::vector<cl_mem> m_devicePtrs;
  bool m_dirty; // needs write to device
};

class XHostMatrixBase {
public:
  XHostMatrixBase() {}
  virtual ~XHostMatrixBase() {}

  virtual void AprepForFPGA(void *data, int rows, int cols, int ld, int blockRows, int blockCols, const double &scale, bool bufALoad) = 0;
  virtual void BprepForFPGA(std::vector<XHostMatrixBase*> &matrixVecOut,
    int blockRows, int blockCols, int bSplit) = 0;
  virtual void BprepForIm2ColFPGA(std::vector<XHostMatrixBase*> &matrixVecOut,
    XBLASIm2ColParams i2cp, const double &scale) = 0;
  virtual void BprepForIm2ColFPGA2(std::vector<XHostMatrixBase*> &matrixVecOut,
    XBLASIm2ColParams i2cp, int blockRows, int blockCols, const double &scale) = 0;
  //virtual void BprepForIm2ColFPGA3(std::vector<XHostMatrixBase*> &matrixVecOut,
  //  XBLASIm2ColParams i2cp, int blockRows, int blockCols, const double &scale) = 0;
  virtual void CprepZeroForFPGA(int rows, int cols, int blockRows, int blockCols) = 0;
  virtual void CpostAfterFPGA(void *C_out, int rows, int cols, int ld, int beta, bool shift16, const double &scale) = 0;
  virtual void CpostAfterIm2ColFPGA(void *C_out, int rows, int cols, int ld, int beta, bool shift16, const double &scale, XBLASIm2ColParams i2cp) = 0;
  virtual void ReshapeCAfterPoolFPGA(int bxPiSrc, int bxPiDst) = 0;
  virtual int getSize() const = 0; 
  virtual void* getAddr() = 0;
  virtual int getNumRows() const = 0;
  virtual int getNumCols() const = 0;
  virtual int getLd() const = 0;
  virtual int getPhysicalSize() const = 0;
  virtual int getNumBlockRows() const = 0;
  virtual int getNumBlockCols() const = 0;
  virtual int getBlockWidth() const = 0;
  virtual int getBlockHeight() const = 0;
  virtual long computeChecksum() const = 0;
};

template <typename HostT, typename T>
class XHostMatrix : public XHostMatrixBase {
public:
  XHostMatrix();
  ~XHostMatrix();
  XHostMatrix(HostT *m, int rows, int cols, int ld, 
    int p_Trans, const double scale);
  XHostMatrix(T *m, int rows, int cols, int ld); // "no copy" matrix
  XHostMatrix(int rows, int cols, int ld, int blocksY=1, int blocksX=1);
  XHostMatrix(const XHostMatrix &src); // copy constructor

  int getNumRows() const { return m_rows; }
  int getNumCols() const { return m_cols; }
  int getLd() const { return m_ld; }
  int getNumBlockRows() const { return m_blocksY; }
  int getNumBlockCols() const { return m_blocksX; }
  int getBlockHeight() const { return m_rows / m_blocksY; }
  int getBlockWidth() const { return m_cols / m_blocksX; }
  int getPhysicalSize() const { return m_rows * m_ld; }
  int getSize() const { return m_rows * m_cols; }
  T getVal(int row, int col) const {
    return m_arr[(row * m_ld) + col];
  }
  void* getAddr() { return(m_arr); }
  long computeChecksum() const;

  void AprepForFPGA(void *data, int rows, int cols, int ld, 
    int blockRows, int blockCols, const double &scale, bool bufALoad);
  void BprepForFPGA(std::vector<XHostMatrixBase*> &matrixVecOut,
    int blockRows, int blockCols, int bSplit);
  void BprepForIm2ColFPGA(std::vector<XHostMatrixBase*> &matrixVecOut,
    XBLASIm2ColParams i2cp, const double &scale);
  void BprepForIm2ColFPGA2(std::vector<XHostMatrixBase*> &matrixVecOut,
    XBLASIm2ColParams i2cp, int blockRows, int blockCols, const double &scale);
  //void BprepForIm2ColFPGA3(std::vector<XHostMatrixBase*> &matrixVecOut,
  //  XBLASIm2ColParams i2cp, int blockRows, int blockCols, const double &scale);
  void CprepZeroForFPGA(int rows, int cols, int blockRows, int blockCols);
  void CpostAfterFPGA(void *C_out, int rows, int cols, int ld, int beta, bool shift16, const double &scale);
  void CpostAfterIm2ColFPGA(void *C_out, int rows, int cols, int ld, int beta, bool shift16, const double &scale, XBLASIm2ColParams i2cp);
  void ReshapeCAfterPoolFPGA(int bxPiSrc, int bxPiDst);
  void dumpGemmC(std::string name);

private:
  bool m_ownArrMem;
  bool m_useMemMgr;
  T *m_arr;
  int m_rows;
  int m_cols;
  int m_ld;
  int m_blocksY;
  int m_blocksX;
};

/******************************************************************* 
 * XBLAS Client API
 *******************************************************************/
int xMalloc(XBLASHandle &handle, XMemPtr *&memPtr, size_t size, bool isMappedToHost=false);
int xMalloc(XBLASHandle &handle, XMemPtr **memPtr, size_t size, bool isMappedToHost=false);
void xFree(XBLASHandle &handle, XMemPtr *memPtr);
//int xMallocHost(XBLASHandle &handle, void *&hostMemPtr, size_t size);
//void xFreeHost(XBLASHandle &handle, void *hostMemPtr);
int xMemcpy(XBLASHandle &handle, void *hostMemPtr, XMemPtr *deviceMemPtr, size_t size);
int xMemcpy(XBLASHandle &handle, XMemPtr *deviceMemPtr, void *hostMemPtr, size_t size);

int xblasCreate(XBLASHandle *&handle, 
  const char *xclbin, const char *kernelName);
void xblasMemset(XBLASHandle &handle, const size_t N, const int alpha, XMemPtr *ptr);
template <typename T>
int xblasSgemm(XBLASHandle &handle, const int blasOrder,
         const int blasTransA, const int blasTransB,
	       const int M, const int N, const int K,
	       const int alpha,
	       XMemPtr *A, const int la,
	       XMemPtr *B, const int lb,
	       const int beta,
	       XMemPtr *C, const int lc, 
         XBLASConfig *cfg=NULL);
template <typename T>
int xblasSgemm(XBLASHandle &handle, const int blasOrder,
         const int blasTransA, const int blasTransB,
	       const int M, const int N, const int K,
	       const T alpha,
	       const T *A, const int la,
	       const T *B, const int lb,
	       const T beta,
	       T *C, const int lc, 
         XBLASConfig *cfg=NULL);

void xblasEnqueueJob(XBLASHandle &handle, XPipelinePacket *pkt);
void xblasDestroy(XBLASHandle *handle, bool printTaskTimes=false);

// special functions to preload matrices to DDR
template <typename HostT, typename T>
void xblasLoadA(XBLASHandle &handle, const int blasTrans, 
  const int M, const int K, const int ld, const HostT *A, XBLASConfig *cfg=NULL);
template <typename HostT, typename T>
void xblasLoadB(XBLASHandle &handle, const int blasTrans, 
  const int N, const int K, const int ld, const HostT *B, XBLASConfig *cfg=NULL);

#endif /*XBLAS_H*/
