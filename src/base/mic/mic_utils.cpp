#include "../../utils/def.hpp"
#include "mic_utils.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "backend_mic.hpp"
#include "mic_allocate_free.hpp"

#include <stdlib.h>


namespace paralution {

template <typename ValueType>
void copy_to_mic(const int mic_dev, const ValueType *src, ValueType *dst, const int size) {

#pragma offload target(mic:mic_dev)	    \
  in(dst:length(0) MIC_REUSE MIC_RETAIN)    \
  in(src:length(size))
#pragma omp parallel for 
	for (int i=0; i<size; ++i) 
        dst[i] = src[i];

}

template <typename ValueType>
void copy_to_host(const int mic_dev, const ValueType *src, ValueType *dst, const int size) {


#pragma offload target(mic:mic_dev)	    \
  in(src:length(0) MIC_REUSE MIC_RETAIN)    \
  out(dst:length(size))
#pragma omp parallel for 
	for (int i=0; i<size; ++i) 
        dst[i] = src[i];

}

template <typename ValueType>
void copy_mic_mic(const int mic_dev, const ValueType *src, ValueType *dst, const int size) {

#pragma offload target(mic:mic_dev)			    \
  in(src:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(dst:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    dst[i] = src[i];


}

template void copy_to_mic<float>(const int mic_dev, const float *src, float *dst, const int size);
template void copy_to_mic<double>(const int mic_dev, const double *src, double *dst, const int size);
template void copy_to_mic<int>(const int mic_dev, const int *src, int *dst, const int size);
template void copy_to_mic<unsigned int>(const int mic_dev, const unsigned int *src, unsigned int *dst, const int size);

template void copy_to_host<double>(const int mic_dev, const double *src, double *dst, const int size);
template void copy_to_host<float>(const int mic_dev, const float *src, float *dst, const int size);
template void copy_to_host<int>(const int mic_dev, const int *src, int *dst, const int size);
template void copy_to_host<unsigned int>(const int mic_dev, const unsigned int *src, unsigned int *dst, const int size);

template void copy_mic_mic<float>(const int mic_dev, const float *src, float *dst, const int size);
template void copy_mic_mic<double>(const int mic_dev, const double *src, double *dst, const int size);
template void copy_mic_mic<int>(const int mic_dev, const int *src, int *dst, const int size);
template void copy_mic_mic<unsigned int>(const int mic_dev, const unsigned int *src, unsigned int *dst, const int size);

};
