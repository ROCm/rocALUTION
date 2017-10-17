#ifndef PARALUTION_MIC_MATRIX_CSR_KERNEL_HPP_
#define PARALUTION_MIC_MATRIX_CSR_KERNEL_HPP_

namespace paralution {

template <typename ValueType>
void spmv_csr(const int mic_dev, 
	      const int *row, const int *col, const ValueType *val,
	      const int nrow,
	      const ValueType *in, ValueType *out);

template <typename ValueType>
void spmv_add_csr(const int mic_dev, 
		  const int *row, const int *col, const ValueType *val,
		  const int nrow,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out);

}

#endif // PARALUTION_BASE_MATRIX_CSR_KERNEL_HPP_
