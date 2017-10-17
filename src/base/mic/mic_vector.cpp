#include "../../utils/def.hpp"
#include "mic_vector.hpp"
#include "../base_vector.hpp"
#include "../host/host_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "mic_utils.hpp"
#include "mic_allocate_free.hpp"
#include "mic_vector_kernel.hpp"



namespace paralution {

template <typename ValueType>
MICAcceleratorVector<ValueType>::MICAcceleratorVector() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
MICAcceleratorVector<ValueType>::MICAcceleratorVector(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "MICAcceleratorVector::MICAcceleratorVector()",
            "constructor with local_backend");

  this->vec_ = NULL;
  this->set_backend(local_backend); 

}


template <typename ValueType>
MICAcceleratorVector<ValueType>::~MICAcceleratorVector() {

  LOG_DEBUG(this, "MICAcceleratorVector::~MICAcceleratorVector()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::info(void) const {

  LOG_INFO("MICAcceleratorVector<ValueType>");

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::Allocate(const int n) {

  assert(n >= 0);

  if (this->get_size() >0)
    this->Clear();

  if (n > 0) {

    allocate_mic(this->local_backend_.MIC_dev,
		 n, &this->vec_);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    n, this->vec_);
    this->size_ = n;
  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::SetDataPtr(ValueType **ptr, const int size) {

  assert(*ptr != NULL);
  assert(size > 0);

  this->vec_ = *ptr;
  this->size_ = size;

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::LeaveDataPtr(ValueType **ptr) {

  assert(this->get_size() > 0);

  *ptr = this->vec_;
  this->vec_ = NULL;

  this->size_ = 0 ;

}


template <typename ValueType>
void MICAcceleratorVector<ValueType>::Clear(void) {
  
  if (this->get_size() >0) {

    free_mic(this->local_backend_.MIC_dev,
	     &this->vec_);

    this->size_ = 0 ;

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyFromHost(const HostVector<ValueType> &src) {

  // CPU to MIC copy
  const HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {

  if (this->get_size() == 0)
    this->Allocate(cast_vec->get_size());
    
    assert(cast_vec->get_size() == this->get_size());

    if (this->get_size() >0) {

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_vec->vec_, this->vec_, this->get_size());

    }

  } else {
    
    LOG_INFO("Error unsupported MIC vector type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}



template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyToHost(HostVector<ValueType> *dst) const {

  // MIC to CPU copy
  HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {

  if (cast_vec->get_size() == 0)
    cast_vec->Allocate(this->get_size());  
    
    assert(cast_vec->get_size() == this->get_size());

    if (this->get_size() >0) {

      copy_to_host(this->local_backend_.MIC_dev,
		   this->vec_, cast_vec->vec_, this->get_size());

    }

  } else {
    
    LOG_INFO("Error unsupported MIC vector type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

  
}


template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src) {

  const MICAcceleratorVector<ValueType> *mic_cast_vec;
  const HostVector<ValueType> *host_cast_vec;


    // MIC to MIC copy
    if ((mic_cast_vec = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&src)) != NULL) {

      if (this->get_size() == 0)
        this->Allocate(mic_cast_vec->get_size());

      assert(mic_cast_vec->get_size() == this->get_size());

      if (this != mic_cast_vec)  {  

        if (this->get_size() >0) {

	  copy_mic_mic(this->local_backend_.MIC_dev,
		       mic_cast_vec->vec_, this->vec_, this->get_size());

        }

      }

    } else {
      
      //MIC to CPU copy
      if ((host_cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {
        

        this->CopyFromHost(*host_cast_vec);
        
      
      } else {

        LOG_INFO("Error unsupported MIC vector type");
        this->info();
        src.info();
        FATAL_ERROR(__FILE__, __LINE__);
        
      }
      
    }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src,
                                               const int src_offset,
                                               const int dst_offset,
                                               const int size) {

  assert(&src != this);
  assert(this->get_size() > 0);
  assert(src.  get_size() > 0);
  assert(size > 0);

  assert(src_offset + size <= src.get_size());
  assert(dst_offset + size <= this->get_size());

  const MICAcceleratorVector<ValueType> *cast_src = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&src);
  assert(cast_src != NULL);

  
  copy_mic_mic(this->local_backend_.MIC_dev,
	       cast_src->vec_+src_offset, 
	       this->vec_+dst_offset, 
	       size);
  
}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyTo(BaseVector<ValueType> *dst) const{

  MICAcceleratorVector<ValueType> *mic_cast_vec;
  HostVector<ValueType> *host_cast_vec;

    // MIC to MIC copy
    if ((mic_cast_vec = dynamic_cast<MICAcceleratorVector<ValueType>*> (dst)) != NULL) {

      if (mic_cast_vec->get_size() == 0)
        mic_cast_vec->Allocate(this->get_size());

      assert(mic_cast_vec->get_size() == this->get_size());

      if (this != mic_cast_vec)  {  

        if (this->get_size() >0) {

	  copy_mic_mic(this->local_backend_.MIC_dev,
		       this->vec_, mic_cast_vec->vec_, this->get_size());

        }
      }

    } else {
      
      //MIC to CPU copy
      if ((host_cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {
        

        this->CopyToHost(host_cast_vec);
        
      
      } else {

        LOG_INFO("Error unsupported MIC vector type");
        this->info();
        dst->info();
        FATAL_ERROR(__FILE__, __LINE__);
        
      }
      
    }


}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::Zeros(void) {

  if (this->get_size() > 0) {

    set_to_zero_mic(this->local_backend_.MIC_dev,
		    this->get_size(), this->vec_);
    
  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::Ones(void) {

  if (this->get_size() > 0)
    set_to_one_mic(this->local_backend_.MIC_dev,
		   this->get_size(), this->vec_);

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::SetValues(const ValueType val) {

  LOG_INFO("MICAcceleratorVector::SetValues NYI");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::AddScale(const BaseVector<ValueType> &x, const ValueType alpha) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    
    const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    addscale(this->local_backend_.MIC_dev,
	     cast_x->vec_, alpha, this->get_size(), this->vec_);

  }

}


template <typename ValueType>
void MICAcceleratorVector<ValueType>::ScaleAdd(const ValueType alpha, const BaseVector<ValueType> &x) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    
    const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    scaleadd(this->local_backend_.MIC_dev,
	     cast_x->vec_, alpha, this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    
    const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    scaleaddscale(this->local_backend_.MIC_dev,
		  cast_x->vec_, alpha, beta, this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta,
                                          const int src_offset, const int dst_offset,const int size) {

  assert(this->get_size() > 0);
  assert(x.    get_size() > 0);
  assert(size > 0);

  assert(src_offset + size <= x.get_size());
  assert(dst_offset + size <= this->get_size());

  const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
  assert(cast_x != NULL);

  scaleaddscale(this->local_backend_.MIC_dev,
		cast_x->vec_, alpha, beta, this->vec_,
		src_offset, dst_offset, size);



}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::ScaleAdd2(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta, const BaseVector<ValueType> &y, const ValueType gamma) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    assert(this->get_size() == y.get_size());
    
    const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
    const MICAcceleratorVector<ValueType> *cast_y = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&y);
    assert(cast_x != NULL);
    assert(cast_y != NULL);

    scaleadd2(this->local_backend_.MIC_dev,
	      cast_x->vec_, cast_y->vec_,
	      alpha, beta, gamma,
	      this->get_size(),
	      this->vec_);
    
  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::Scale(const ValueType alpha) {

  if (this->get_size() > 0) {

    scale(this->local_backend_.MIC_dev,
	  alpha, this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::ExclusiveScan(const BaseVector<ValueType> &x) {

  LOG_INFO("MICAcceleratorVector::ExclusiveScan() NYI");
  FATAL_ERROR(__FILE__, __LINE__); 

}

template <typename ValueType>
ValueType MICAcceleratorVector<ValueType>::Dot(const BaseVector<ValueType> &x) const {

  assert(this->get_size() == x.get_size());

  const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
  assert(cast_x != NULL);

  ValueType d;

  dot(this->local_backend_.MIC_dev,
      this->vec_, cast_x->vec_, this->get_size(), d);

  return d;
}

template <typename ValueType>
ValueType MICAcceleratorVector<ValueType>::DotNonConj(const BaseVector<ValueType> &x) const {

  return this->Dot(x);
}


template <typename ValueType>
ValueType MICAcceleratorVector<ValueType>::Asum(void) const {

  ValueType asumv;

  asum(this->local_backend_.MIC_dev,
       this->vec_, this->get_size(), asumv);

  return asumv;

}

template <typename ValueType>
int MICAcceleratorVector<ValueType>::Amax(ValueType &value) const {

  int index = 0;

  amax(this->local_backend_.MIC_dev,
       this->vec_, this->get_size(), value, index);

  return index;

}


template <typename ValueType>
ValueType MICAcceleratorVector<ValueType>::Norm(void) const {

  ValueType n;

  norm(this->local_backend_.MIC_dev,
       this->vec_, this->get_size(), n);

  return n;

}


template <typename ValueType>
ValueType MICAcceleratorVector<ValueType>::Reduce(void) const {

  ValueType r = ValueType(0.0);

  reduce(this->local_backend_.MIC_dev,
	 this->vec_, this->get_size(), r);

  return r;

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    
    const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    pointwisemult(this->local_backend_.MIC_dev,
		  cast_x->vec_, this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x, const BaseVector<ValueType> &y) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    assert(this->get_size() == y.get_size());
    
    const MICAcceleratorVector<ValueType> *cast_x = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&x);
    const MICAcceleratorVector<ValueType> *cast_y = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&y);
    assert(cast_x != NULL);
    assert(cast_y != NULL);

    pointwisemult2(this->local_backend_.MIC_dev,
		   cast_x->vec_, cast_y->vec_, this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::Permute(const BaseVector<int> &permutation) {

  if (this->get_size() > 0) {

    assert(&permutation != NULL);
    assert(this->get_size() == permutation.get_size());
    
    const MICAcceleratorVector<int> *cast_perm = dynamic_cast<const MICAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);
    
    MICAcceleratorVector<ValueType> vec_tmp(this->local_backend_);     
    vec_tmp.Allocate(this->get_size());
    vec_tmp.CopyFrom(*this);

    permute(this->local_backend_.MIC_dev,
	    cast_perm->vec_, vec_tmp.vec_,
	    this->get_size(), this->vec_);

  }
}


template <typename ValueType>
void MICAcceleratorVector<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  if (this->get_size() > 0) {

    assert(&permutation != NULL);
    assert(this->get_size() == permutation.get_size());
    
    const MICAcceleratorVector<int> *cast_perm = dynamic_cast<const MICAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);
    
    MICAcceleratorVector<ValueType> vec_tmp(this->local_backend_);   
    vec_tmp.Allocate(this->get_size());
    vec_tmp.CopyFrom(*this);

    permuteback(this->local_backend_.MIC_dev,
		cast_perm->vec_, vec_tmp.vec_,
		this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyFromPermute(const BaseVector<ValueType> &src,
                                                      const BaseVector<int> &permutation) { 

  if (this->get_size() > 0) {

    assert(this != &src);
    
    const MICAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&src);
    const MICAcceleratorVector<int> *cast_perm      = dynamic_cast<const MICAcceleratorVector<int>*> (&permutation) ; 
    assert(cast_perm != NULL);
    assert(cast_vec  != NULL);
    
    assert(cast_vec ->get_size() == this->get_size());
    assert(cast_perm->get_size() == this->get_size());

    permute(this->local_backend_.MIC_dev,
	    cast_perm->vec_, cast_vec->vec_,
	    this->get_size(),
	    this->vec_);
 
  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType> &src,
                                                              const BaseVector<int> &permutation) {

  if (this->get_size() > 0) {

    assert(this != &src);
    
    const MICAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&src);
    const MICAcceleratorVector<int> *cast_perm      = dynamic_cast<const MICAcceleratorVector<int>*> (&permutation) ; 
    assert(cast_perm != NULL);
    assert(cast_vec  != NULL);
    
    assert(cast_vec ->get_size() == this->get_size());
    assert(cast_perm->get_size() == this->get_size());

    permuteback(this->local_backend_.MIC_dev,
		cast_perm->vec_, cast_vec->vec_,
		this->get_size(),
		this->vec_);
        
  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::Power(const double val) {

  if (this->get_size() > 0) {
    
    power(this->local_backend_.MIC_dev,
          this->get_size(), val, this->vec_);
  
  }

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::ExtractCoarseMapping(const int start, const int end, const int *index,
                                                           const int nc, int *size, int *map) const {

  LOG_INFO("ExtractCoarseMapping() NYI for MIC");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void MICAcceleratorVector<ValueType>::ExtractCoarseBoundary(const int start, const int end, const int *index,
                                                            const int nc, int *size, int *boundary) const {

  LOG_INFO("ExtractCoarseBoundary() NYI for MIC");
  FATAL_ERROR(__FILE__, __LINE__);

}


template class MICAcceleratorVector<double>;
template class MICAcceleratorVector<float>;

template class MICAcceleratorVector<int>;

}
