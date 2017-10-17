#ifndef PARALUTION_MIC_VECTOR_KERNEL_HPP_
#define PARALUTION_MIC_VECTOR_KERNEL_HPP_


namespace paralution {

template <typename ValueType>
void dot(const int mic_dev, const ValueType *vec1, const ValueType *vec2, const int size, ValueType &d);

template <typename ValueType>
void asum(const int mic_dev, const ValueType *vec, const int size, ValueType &d);

template <typename ValueType>
void amax(const int mic_dev, const ValueType *vec, const int size, ValueType &d, int &index);

template <typename ValueType>
void norm(const int mic_dev, const ValueType *vec, const int size, ValueType &d);

template <typename ValueType>
void reduce(const int mic_dev, const ValueType *vec, const int size, ValueType &d);

template <typename ValueType>
void scaleadd(const int mic_dev, const ValueType *vec1, const ValueType alpha, const int size, ValueType *vec2);

template <typename ValueType>
void addscale(const int mic_dev, const ValueType *vec1, const ValueType alpha, const int size, ValueType *vec2);

template <typename ValueType>
void scaleaddscale(const int mic_dev, 
		   const ValueType *vec1, const ValueType alpha, const ValueType beta, 
		   const int size, ValueType *vec2);

template <typename ValueType>
void scaleaddscale(const int mic_dev, 
		   const ValueType *vec1, const ValueType alpha, 
		   const ValueType beta, ValueType *vec2,
		   const int src_offset, const int dst_offset,const int size);

template <typename ValueType>
void scaleadd2(const int mic_dev, 
	       const ValueType *vec1, const ValueType *vec2, 
	       const ValueType alpha, const ValueType beta, const ValueType gamma,
	       const int size, ValueType *vec3);

template <typename ValueType>
void scale(const int mic_dev, 
	   const ValueType alpha, const int size, ValueType *vec);

template <typename ValueType>
void pointwisemult(const int mic_dev, 
		   const ValueType *vec1, const int size, ValueType *vec2);

template <typename ValueType>
void pointwisemult2(const int mic_dev, 
		    const ValueType *vec1,  const ValueType *vec2, 
		    const int size, ValueType *vec3);

template <typename ValueType>
void permute(const int mic_dev, 
	     const int *perm, const ValueType *in, 
	     const int size, ValueType *out);

template <typename ValueType>
void permuteback(const int mic_dev, 
		 const int *perm, const ValueType *in, 
		 const int size, ValueType *out);

template <typename ValueType>
void power(const int mic_dev, 
           const int size, const double val, ValueType *vec);

}

#endif // PARALUTION_BASE_VECTOR_KERNEL_HPP_
