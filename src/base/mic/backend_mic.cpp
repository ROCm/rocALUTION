#include "../../utils/def.hpp"
#include "../backend_manager.hpp"
#include "backend_mic.hpp" 
#include "../../utils/log.hpp"
#include "mic_utils.hpp"
#include "../base_vector.hpp"
#include "../base_matrix.hpp"

#include "mic_vector.hpp"
#include "mic_matrix_csr.hpp"
#include "mic_matrix_coo.hpp"
#include "mic_matrix_mcsr.hpp"
#include "mic_matrix_bcsr.hpp"
#include "mic_matrix_hyb.hpp"
#include "mic_matrix_dia.hpp"
#include "mic_matrix_ell.hpp"
#include "mic_matrix_dense.hpp"


namespace paralution {

bool paralution_init_mic(void) {

  LOG_DEBUG(0, "paralution_init_mic()",
            "* begin");

  // no nothing
  LOG_INFO("MIC backed is initialized");

  LOG_DEBUG(0, "paralution_init_mic()",
            "* end");


  return true;

}


void paralution_stop_mic(void) {

  LOG_DEBUG(0, "paralution_stop_mic()",
            "* begin");

  if (_get_backend_descriptor()->accelerator) {

    // no nothing

  }

  LOG_DEBUG(0, "paralution_stop_mic()",
            "* end");

}

void paralution_info_mic(const struct Paralution_Backend_Descriptor backend_descriptor) {

  int num_dev = 0 ;
  num_dev = _Offload_number_of_devices();

  LOG_INFO("Number of MIC devices in the system: " << num_dev);
  LOG_INFO("Selected MIC devices:" << backend_descriptor.MIC_dev);

}

template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_mic_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format) {
  assert(backend_descriptor.backend == MIC);

  switch (matrix_format) {

  case CSR:
    return new MICAcceleratorMatrixCSR<ValueType>(backend_descriptor);
    
  case COO:
    return new MICAcceleratorMatrixCOO<ValueType>(backend_descriptor);

  case MCSR:
    return new MICAcceleratorMatrixMCSR<ValueType>(backend_descriptor);

  case DIA:
    return new MICAcceleratorMatrixDIA<ValueType>(backend_descriptor);
    
  case ELL:
    return new MICAcceleratorMatrixELL<ValueType>(backend_descriptor);

  case DENSE:
    return new MICAcceleratorMatrixDENSE<ValueType>(backend_descriptor);

  case HYB:
    return new MICAcceleratorMatrixHYB<ValueType>(backend_descriptor);

  case BCSR:
    return new MICAcceleratorMatrixBCSR<ValueType>(backend_descriptor);


      
  default:
    LOG_INFO("This backed is not supported for Matrix types");
    FATAL_ERROR(__FILE__, __LINE__);   
    return NULL;
  } 

}


template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_mic_vector(const struct Paralution_Backend_Descriptor backend_descriptor) {

  assert(backend_descriptor.backend == MIC);

  return new MICAcceleratorVector<ValueType>(backend_descriptor);

}


template AcceleratorVector<float>* _paralution_init_base_mic_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<double>* _paralution_init_base_mic_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<int>* _paralution_init_base_mic_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

template AcceleratorMatrix<float>* _paralution_init_base_mic_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                         const unsigned int matrix_format);
template AcceleratorMatrix<double>* _paralution_init_base_mic_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                          const unsigned int matrix_format);


};
