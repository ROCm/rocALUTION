#include "../../utils/def.hpp"
#include "mic_vector_kernel.hpp"
#include "mic_utils.hpp"
#include <math.h>


namespace paralution {

template <typename ValueType>
void dot(const int mic_dev, const ValueType *vec1, const ValueType *vec2, const int size, ValueType &d) {
  
  ValueType tmp = ValueType(0.0);

#pragma offload target(mic:mic_dev)			    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for reduction(+:tmp)
  for (int i=0; i<size; ++i)
    tmp += vec1[i]*vec2[i];

  d = tmp;

}

template <typename ValueType>
void asum(const int mic_dev, const ValueType *vec, const int size, ValueType &d) {
  
  ValueType tmp = ValueType(0.0);

#pragma offload target(mic:mic_dev)			    \
  in(vec:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for reduction(+:tmp)
  for (int i=0; i<size; ++i)
    tmp += fabs(vec[i]);

  d = tmp;

}


template <typename ValueType>
void amax(const int mic_dev, const ValueType *vec, const int size, ValueType &d, int &index) {
  
  int ind = 0;
  ValueType tmp = ValueType(0.0);

#pragma offload target(mic:mic_dev)			    \
  in(vec:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for
  for (int i=0; i<size; ++i) {
    ValueType val = fabs(vec[i]);
    if (val > tmp)
#pragma omp critical
      {
	if (val > tmp) {
	  tmp = val;
	  ind = i;
	}
      }
    }

  d = tmp;
  index = ind;

}


template <typename ValueType>
void norm(const int mic_dev, const ValueType *vec, const int size, ValueType &d) {
  
  ValueType tmp = ValueType(0.0);

#pragma offload target(mic:mic_dev)			    \
  in(vec:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for reduction(+:tmp)
  for (int i=0; i<size; ++i)
    tmp += vec[i]*vec[i];

  d = sqrt(tmp);

}

template <typename ValueType>
void reduce(const int mic_dev, const ValueType *vec, const int size, ValueType &d) {
  
  ValueType tmp = ValueType(0.0);

#pragma offload target(mic:mic_dev)			    \
  in(vec:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for reduction(+:tmp)
  for (int i=0; i<size; ++i)
    tmp += vec[i];

  d = tmp;

}


template <typename ValueType>
void scaleadd(const int mic_dev, const ValueType *vec1, const ValueType alpha, const int size, ValueType *vec2) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec2[i] = alpha*vec2[i] + vec1[i];

}

template <typename ValueType>
void addscale(const int mic_dev, const ValueType *vec1, const ValueType alpha, const int size, ValueType *vec2) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec2[i] = vec2[i] + alpha*vec1[i];

}


template <typename ValueType>
void scaleaddscale(const int mic_dev, 
		   const ValueType *vec1, const ValueType alpha, const ValueType beta, 
		   const int size, ValueType *vec2) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec2[i] = alpha*vec2[i] + beta*vec1[i];

}

template <typename ValueType>
void scaleaddscale(const int mic_dev, 
		   const ValueType *vec1, const ValueType alpha, 
		   const ValueType beta, ValueType *vec2,
		   const int src_offset, const int dst_offset,const int size) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec2[i+dst_offset] = alpha*vec2[i+dst_offset] + beta*vec1[i+src_offset];


}

template <typename ValueType>
void scaleadd2(const int mic_dev, 
	       const ValueType *vec1, const ValueType *vec2, 
	       const ValueType alpha, const ValueType beta, const ValueType gamma,
	       const int size, ValueType *vec3) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec3:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec3[i] = alpha*vec3[i] + beta*vec1[i] + gamma*vec2[i];

}

template <typename ValueType>
void scale(const int mic_dev, 
	   const ValueType alpha, const int size, ValueType *vec) {

#pragma offload target(mic:mic_dev)				    \
  in(vec:length(0) MIC_REUSE MIC_RETAIN)		    
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec[i] *= alpha;
  
}

template <typename ValueType>
void pointwisemult(const int mic_dev, 
		   const ValueType *vec1, const int size, ValueType *vec2) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec2[i] = vec2[i]*vec1[i];

}

template <typename ValueType>
void pointwisemult2(const int mic_dev, 
		    const ValueType *vec1,  const ValueType *vec2, 
		    const int size, ValueType *vec3) {

#pragma offload target(mic:mic_dev)				    \
  in(vec1:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec2:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(vec3:length(0) MIC_REUSE MIC_RETAIN)		    
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec3[i] = vec2[i]*vec1[i];

}

template <typename ValueType>
void permute(const int mic_dev, 
	     const int *perm, const ValueType *in, 
	     const int size, ValueType *out) {
  
#pragma offload target(mic:mic_dev)				    \
  in(perm:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)			    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)		    
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    out[ perm[i] ] = in[i];
  
}

template <typename ValueType>
void permuteback(const int mic_dev, 
		 const int *perm, const ValueType *in, 
		 const int size, ValueType *out) {
  
#pragma offload target(mic:mic_dev)				    \
  in(perm:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)			    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)		    
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    out[i] = in[ perm[i] ];
  
}

template <typename ValueType>
void power(const int mic_dev, 
           const int size, const double val, ValueType *vec) {

#pragma offload target(mic:mic_dev)				    \
  in(vec:length(0) MIC_REUSE MIC_RETAIN) 
#pragma omp parallel for 
  for (int i=0; i<size; ++i)
    vec[i] = pow(vec[i], ValueType(val));

}

// double
template void dot<double>(const int mic_dev, const double *vec1, const double *vec2, const int size, double &d);

template void asum<double>(const int mic_dev, const double *vec, const int size, double &d);

template void amax<double>(const int mic_dev, const double *vec, const int size, double &d, int &index);

template void norm<double>(const int mic_dev, const double *vec, const int size, double &d);

template void reduce<double>(const int mic_dev, const double *vec, const int size, double &d);

template void scaleadd<double>(const int mic_dev, const double *vec1, const double alpha, const int size, double *vec2);

template void addscale<double>(const int mic_dev, const double *vec1, const double alpha, const int size, double *vec2);

template void scaleaddscale<double>(const int mic_dev, 
				    const double *vec1, const double alpha, const double beta, 
				    const int size, double *vec2);
  
template void scaleaddscale<double>(const int mic_dev, 
				    const double *vec1, const double alpha, 
				    const double beta, double *vec2,
				    const int src_offset, const int dst_offset,const int size);

template void scaleadd2<double>(const int mic_dev, 
				const double *vec1, const double *vec2, 
				const double alpha, const double beta, const double gamma,
				const int size, double *vec3);

template void scale<double>(const int mic_dev, 
			    const double alpha, const int size, double *vec);

template void pointwisemult<double>(const int mic_dev, 
				    const double *vec1, const int size, double *vec2);

template void pointwisemult2<double>(const int mic_dev, 
				     const double *vec1,  const double *vec2, 
				     const int size, double *vec3);

template void permute<double>(const int mic_dev, 
			      const int *perm, const double *in, 
			      const int size, double *out);
  
template void permuteback<double>(const int mic_dev, 
				  const int *perm, const double *in, 
				  const int size, double *out);

template void power<double>(const int mic_dev, 
			    const int size, const double val, double *vec);


// float
template void dot<float>(const int mic_dev, const float *vec1, const float *vec2, const int size, float &d);

template void asum<float>(const int mic_dev, const float *vec, const int size, float &d);

template void amax<float>(const int mic_dev, const float *vec, const int size, float &d, int &index);

template void norm<float>(const int mic_dev, const float *vec, const int size, float &d);

template void reduce<float>(const int mic_dev, const float *vec, const int size, float &d);

template void scaleadd<float>(const int mic_dev, const float *vec1, const float alpha, const int size, float *vec2);

template void addscale<float>(const int mic_dev, const float *vec1, const float alpha, const int size, float *vec2);

template void scaleaddscale<float>(const int mic_dev, 
				   const float *vec1, const float alpha, const float beta, 
				   const int size, float *vec2);
  
template void scaleaddscale<float>(const int mic_dev, 
				   const float *vec1, const float alpha, 
				   const float beta, float *vec2,
				   const int src_offset, const int dst_offset,const int size);
  
template void scaleadd2<float>(const int mic_dev, 
			       const float *vec1, const float *vec2, 
			       const float alpha, const float beta, const float gamma,
			       const int size, float *vec3);

template void scale<float>(const int mic_dev, 
			   const float alpha, const int size, float *vec);

template void pointwisemult<float>(const int mic_dev, 
				   const float *vec1, const int size, float *vec2);

template void pointwisemult2<float>(const int mic_dev, 
				    const float *vec1,  const float *vec2, 
				    const int size, float *vec3);

template void permute<float>(const int mic_dev, 
			     const int *perm, const float *in, 
			     const int size, float *out);
  
template void permuteback<float>(const int mic_dev, 
				 const int *perm, const float *in, 
				 const int size, float *out);

template void power<float>(const int mic_dev, 
			   const int size, const double val, float *vec);


// int

template void dot<int>(const int mic_dev, const int *vec1, const int *vec2, const int size, int &d);

template void asum<int>(const int mic_dev, const int *vec, const int size, int &d);

template void amax<int>(const int mic_dev, const int *vec, const int size, int &d, int &index);

template void norm<int>(const int mic_dev, const int *vec, const int size, int &d);

template void reduce<int>(const int mic_dev, const int *vec, const int size, int &d);

template void scaleadd<int>(const int mic_dev, const int *vec1, const int alpha, const int size, int *vec2);

template void addscale<int>(const int mic_dev, const int *vec1, const int alpha, const int size, int *vec2);

template void scaleaddscale<int>(const int mic_dev, 
				 const int *vec1, const int alpha, const int beta, 
				 const int size, int *vec2);

template void scaleaddscale<int>(const int mic_dev, 
				 const int *vec1, const int alpha, 
				 const int beta, int *vec2,
				 const int src_offset, const int dst_offset,const int size);

template void scaleadd2<int>(const int mic_dev, 
			     const int *vec1, const int *vec2, 
			     const int alpha, const int beta, const int gamma,
			     const int size, int *vec3);

template void scale<int>(const int mic_dev, 
			 const int alpha, const int size, int *vec);

template void pointwisemult<int>(const int mic_dev, 
				 const int *vec1, const int size, int *vec2);

template void pointwisemult2<int>(const int mic_dev, 
				  const int *vec1,  const int *vec2, 
				  const int size, int *vec3);

template void permute<int>(const int mic_dev, 
			   const int *perm, const int *in, 
			   const int size, int *out);

template void permuteback<int>(const int mic_dev, 
			       const int *perm, const int *in, 
			       const int size, int *out);

template void power<int>(const int mic_dev, 
			 const int size, const double val, int *vec);


}

