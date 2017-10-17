#include "../../utils/def.hpp"
#include "mic_matrix_dia_kernel.hpp"
#include "mic_utils.hpp"
#include "../matrix_formats_ind.hpp"

namespace paralution {

template <typename ValueType>
void spmv_dia(const int mic_dev, 
	      const int *offset,  const ValueType *val,
	      const int nrow,
	      const int ndiag,
	      const ValueType *in, ValueType *out) {
  

#pragma offload target(mic:mic_dev)			    \
  in(offset:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(val:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for
    for (int i=0; i<nrow; ++i) {
      
      out[i] = ValueType(0.0);
      
      for (int j=0; j<ndiag; ++j) {
        
        int start  = 0;
        int end = nrow;
        int v_offset = 0; 
        
        if ( offset[j] < 0) {
          start -= offset[j];
          v_offset = -start;
        } else {
          end -= offset[j];
          v_offset = offset[j];
        }
        
        if ( (i >= start) && (i < end)) {
          out[i] += val[DIA_IND(i, j, nrow, ndiag)] * in[i+v_offset];
        } else {
          if (i >= end)
            break;
	}
      }
    }

}

template <typename ValueType>
void spmv_add_dia(const int mic_dev, 
		  const int *offset,  const ValueType *val,
		  const int nrow,
		  const int ndiag,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out) {
  

#pragma offload target(mic:mic_dev)			    \
  in(offset:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(val:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for
    for (int i=0; i<nrow; ++i) {
      
      for (int j=0; j<ndiag; ++j) {
        
        int start  = 0;
        int end = nrow;
        int v_offset = 0; 
        
        if ( offset[j] < 0) {
          start -= offset[j];
          v_offset = -start;
        } else {
          end -= offset[j];
          v_offset = offset[j];
        }
        
        if ( (i >= start) && (i < end)) {
          out[i] += scalar*val[DIA_IND(i, j, nrow, ndiag)] * in[i+v_offset];
        } else {
          if (i >= end)
            break;
	}
      }
    }

}

template void spmv_dia<double>(const int mic_dev, 
				  const int *offset,  const double *val,
				  const int nrow,
				  const int ndiag,
				  const double *in, double *out);

template void spmv_dia<float>(const int mic_dev, 
				  const int *offset,  const float *val,
				  const int nrow,
				  const int ndiag,
				  const float *in, float *out);

template void spmv_dia<int>(const int mic_dev, 
				  const int *offset,  const int *val,
				  const int nrow,
				  const int ndiag,
				  const int *in, int *out);


template void spmv_add_dia<double>(const int mic_dev, 
				     const int *offset,  const double *val,
				     const int nrow,
				     const int ndiag,
				     const double scalar,
				     const double *in, double *out);

template void spmv_add_dia<float>(const int mic_dev, 
				     const int *offset,  const float *val,
				     const int nrow,
				     const int ndiag,
				     const float scalar,
				     const float *in, float *out);

template void spmv_add_dia<int>(const int mic_dev, 
				     const int *offset,  const int *val,
				     const int nrow,
				     const int ndiag,
				     const int scalar,
				     const int *in, int *out);

}


