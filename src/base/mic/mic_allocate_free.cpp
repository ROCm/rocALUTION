#include "../../utils/def.hpp"
#include "mic_allocate_free.hpp"
#include "mic_utils.hpp"
#include "../../utils/log.hpp"

namespace paralution {

template <typename DataType>
void allocate_mic(const int mic_dev, 
		  const int size, DataType **ptr) {

  LOG_DEBUG(0, "allocate_mic()",
            size << " mic_dev=" << mic_dev);

  if (size > 0) {
    assert(*ptr == NULL);

    *ptr = new DataType[size];

    DataType *p = *ptr;

#pragma offload_transfer target(mic:mic_dev)              \
  nocopy(p:length(size) MIC_ALLOC MIC_RETAIN)       

#pragma offload_transfer target(mic:mic_dev)              \
  in(p:length(size) MIC_REUSE MIC_RETAIN) 

    assert(*ptr != NULL);
  }

}

template <typename DataType>
void free_mic(const int mic_dev, 
	      DataType **ptr) {

  LOG_DEBUG(0, "free_mic()",
            *ptr << " mic_dev=" << mic_dev);

  assert(*ptr != NULL);

    DataType *p = *ptr;

#pragma offload_transfer target(mic:mic_dev)              \
  out(p:length(0) MIC_REUSE MIC_FREE) 
  
  delete[] *ptr;

  *ptr = NULL;

}

template <typename DataType>
void set_to_zero_mic(const int mic_dev, 
		     const int size, DataType *ptr) {

  LOG_DEBUG(0, "set_to_zero_mic()",
            "size =" << size << 
            " ptr=" << ptr << " mic_dev=" << mic_dev);

  if (size > 0) {

#pragma offload target(mic:mic_dev)                   \
  in(ptr:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
    for (int i=0; i<size; ++i)
      ptr[i] = DataType(0);

  }

}


template <typename DataType>
void set_to_one_mic(const int mic_dev, 
		    const int size, DataType *ptr) {

  LOG_DEBUG(0, "set_to_one_mic()",
            "size =" << size << 
            " ptr=" << ptr << " mic_dev=" << mic_dev);

  if (size > 0) {

#pragma offload target(mic:mic_dev)                   \
  in(ptr:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
    for (int i=0; i<size; ++i)
      ptr[i] = DataType(1);
    
  }
}



template void allocate_mic<float       >(const int mic_dev, const int size, float        **ptr);
template void allocate_mic<double      >(const int mic_dev, const int size, double       **ptr);
template void allocate_mic<int         >(const int mic_dev, const int size, int          **ptr);
template void allocate_mic<unsigned int>(const int mic_dev, const int size, unsigned int **ptr);
template void allocate_mic<char        >(const int mic_dev, const int size, char         **ptr);

template void free_mic<float       >(const int mic_dev, float        **ptr);
template void free_mic<double      >(const int mic_dev, double       **ptr);
template void free_mic<int         >(const int mic_dev, int          **ptr);
template void free_mic<unsigned int>(const int mic_dev, unsigned int **ptr);
template void free_mic<char        >(const int mic_dev, char         **ptr);

template void set_to_zero_mic<float       >(const int mic_dev, const int size, float        *ptr);
template void set_to_zero_mic<double      >(const int mic_dev, const int size, double       *ptr);
template void set_to_zero_mic<int         >(const int mic_dev, const int size, int          *ptr);
template void set_to_zero_mic<unsigned int>(const int mic_dev, const int size, unsigned int *ptr);
template void set_to_zero_mic<char        >(const int mic_dev, const int size, char         *ptr);


template void set_to_one_mic<float       >(const int mic_dev, const int size, float        *ptr);
template void set_to_one_mic<double      >(const int mic_dev, const int size, double       *ptr);
template void set_to_one_mic<int         >(const int mic_dev, const int size, int          *ptr);
template void set_to_one_mic<unsigned int>(const int mic_dev, const int size, unsigned int *ptr);
template void set_to_one_mic<char        >(const int mic_dev, const int size, char         *ptr);


};


