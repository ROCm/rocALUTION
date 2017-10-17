#ifndef PARALUTION_MIC_MATRIX_DIA_KERNEL_HPP_
#define PARALUTION_MIC_MATRIX_DIA_KERNEL_HPP_

namespace paralution {

template <typename ValueType>
void spmv_dia(const int mic_dev, 
	      const int *offset,  const ValueType *val,
	      const int nrow,
	      const int ndiag,
	      const ValueType *in, ValueType *out);

template <typename ValueType>
void spmv_add_dia(const int mic_dev, 
		  const int *offset,  const ValueType *val,
		  const int nrow,
		  const int ndiag,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out);

}

#endif // PARALUTION_BASE_MATRIX_DIA_KERNEL_HPP_
