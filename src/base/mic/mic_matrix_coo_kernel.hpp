#ifndef PARALUTION_MIC_MATRIX_COO_KERNEL_HPP_
#define PARALUTION_MIC_MATRIX_COO_KERNEL_HPP_

namespace paralution {

template <typename ValueType>
void spmv_coo(const int mic_dev, 
	      const int *row, const int *col, const ValueType *val,
	      const int nrow,
	      const int nnz,
	      const ValueType *in, ValueType *out);
template <typename ValueType>
void spmv_add_coo(const int mic_dev, 
		  const int *row, const int *col, const ValueType *val,
		  const int nrow,
		  const int nnz,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out);

}

#endif // PARALUTION_BASE_MATRIX_COO_KERNEL_HPP_
