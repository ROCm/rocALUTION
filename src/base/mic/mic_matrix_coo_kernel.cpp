#include "../../utils/def.hpp"
#include "mic_matrix_coo_kernel.hpp"
#include "mic_utils.hpp"
#include "../matrix_formats_ind.hpp"

namespace paralution {

template <typename ValueType>
void spmv_coo(const int mic_dev, 
	      const int *row, const int *col, const ValueType *val,
	      const int nrow,
	      const int nnz,
	      const ValueType *in, ValueType *out) {
  
#pragma offload target(mic:mic_dev)			    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for      
  for (int i=0; i<nrow; ++i)
    out[i] = ValueType(0.0);  


#pragma offload target(mic:mic_dev)			    \
  in(row:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(col:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(val:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	    
  for (int i=0; i<nnz; ++i)
    out[row[i]] += val[i] * in[col[i]];
  

}

template <typename ValueType>
void spmv_add_coo(const int mic_dev, 
		  const int *row, const int *col, const ValueType *val,
		  const int nrow,
		  const int nnz,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out) {
  

#pragma offload target(mic:mic_dev)			    \
  in(row:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(col:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(val:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	    
  for (int i=0; i<nnz; ++i)
    out[row[i]] += val[i] * in[col[i]];
  

}

template void spmv_coo<double>(const int mic_dev, 
			       const int *row, const int *col, const double *val,
			       const int nrow,
			       const int nnz,
			       const double *in, double *out);

template void spmv_coo<float>(const int mic_dev, 
			      const int *row, const int *col, const float *val,
			      const int nrow,
			      const int nnz,
			      const float *in, float *out);

template void spmv_coo<int>(const int mic_dev, 
			    const int *row, const int *col, const int *val,
			    const int nrow,
			    const int nnz,
			    const int *in, int *out);

template void spmv_add_coo<double>(const int mic_dev, 
				   const int *row, const int *col, const double *val,
				   const int nrow,
				   const int nnz,
				   const double scalar,
				       const double *in, double *out);

template void spmv_add_coo<float>(const int mic_dev, 
				  const int *row, const int *col, const float *val,
				  const int nrow,
				  const int nnz,
				  const float scalar,
				  const float *in, float *out);

template void spmv_add_coo<int>(const int mic_dev, 
				const int *row, const int *col, const int *val,
				const int nrow,
				const int nnz,
				const int scalar,
				const int *in, int *out);

}

