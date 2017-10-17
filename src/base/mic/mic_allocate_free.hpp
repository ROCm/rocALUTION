#ifndef PARALUTION_MIC_ALLOCATE_FREE_HPP_
#define PARALUTION_MIC_ALLOCATE_FREE_HPP_

namespace paralution {

template <typename DataType>
void allocate_mic(const int mic_dev, const int size, DataType **ptr);

template <typename DataType>
void free_mic(const int mic_dev, DataType **ptr);

template <typename DataType>
void set_to_zero_mic(const int mic_dev, const int size, DataType *ptr);

template <typename DataType>
void set_to_one_mic(const int mic_dev, const int size, DataType *ptr);


};

#endif // PARALUTION_MIC_ALLOCATE_FREE_HPP_



