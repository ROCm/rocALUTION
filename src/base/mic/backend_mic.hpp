#ifndef PARALUTION_BACKEND_MIC_HPP_
#define PARALUTION_BACKEND_MIC_HPP_

#include "../backend_manager.hpp"
#include <iostream>


namespace paralution {

template <typename ValueType>
class AcceleratorVector;
template <typename ValueType>
class AcceleratorMatrix;
template <typename ValueType>
class HostMatrix;

/// Initialize a MIC
bool paralution_init_mic();

/// Release the MIC accelerator
void paralution_stop_mic();

/// Print information about the MICs in the systems
void paralution_info_mic(const struct Paralution_Backend_Descriptor);


/// Build (and return) a vector on MIC
template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_mic_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

/// Build (and return) a matrix on MIC
template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_mic_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format);

};

#endif // PARALUTION_BACKEND_MIC_HPP_
