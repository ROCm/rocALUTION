#ifndef PARALUTION_MIC_MIC_UTILS_HPP_
#define PARALUTION_MIC_MIC_UTILS_HPP_

#define MIC_ALLOC   alloc_if(1)
#define MIC_FREE    free_if(1)
#define MIC_RETAIN  free_if(0)
#define MIC_REUSE   alloc_if(0)

namespace paralution {

template <typename ValueType>
void copy_to_mic(const int mic_dev, const ValueType *src, ValueType *dst, const int size);

template <typename ValueType>
void copy_to_host(const int mic_dev, const ValueType *src, ValueType *dst, const int size);

template <typename ValueType>
void copy_mic_mic(const int mic_dev, const ValueType *src, ValueType *dst, const int size);


};


#endif // PARALUTION_MIC_MIC_UTILS_HPP_
