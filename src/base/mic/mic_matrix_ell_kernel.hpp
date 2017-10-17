#ifndef PARALUTION_MIC_MATRIX_ELL_KERNEL_HPP_
#define PARALUTION_MIC_MATRIX_ELL_KERNEL_HPP_

namespace paralution {

template <typename ValueType>
void spmv_ell(const int mic_dev, 
	      const int *col, const ValueType *val,
	      const int nrow,
	      const int ncol,
	      const int max_row,
	      const ValueType *in, ValueType *out);

template <typename ValueType>
void spmv_add_ell(const int mic_dev, 
		  const int *col, const ValueType *val,
		  const int nrow,
		  const int ncol,
		  const int max_row,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out);

}

#endif // PARALUTION_BASE_MATRIX_ELL_KERNEL_HPP_
