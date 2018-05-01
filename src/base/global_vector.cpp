#include "../utils/def.hpp"
#include "global_vector.hpp"
#include "local_vector.hpp"
#include "../utils/log.hpp"
#include "../utils/allocate_free.hpp"

#ifdef SUPPORT_MULTINODE
#include "../utils/log_mpi.hpp"
#include "../utils/communicator.hpp"
#endif

#include <math.h>
#include <sstream>
#include <limits>
#include <algorithm>
#include <complex>

namespace rocalution {

template <typename ValueType>
GlobalVector<ValueType>::GlobalVector() {

  LOG_DEBUG(this, "GlobalVector::GlobalVector()",
            "default constructor");

#ifndef SUPPORT_MULTINODE
  LOG_INFO("Multinode support disabled");
  FATAL_ERROR(__FILE__, __LINE__);
#endif

  this->object_name_ = "";

  this->recv_boundary_ = NULL;
  this->send_boundary_ = NULL;

}

template <typename ValueType>
GlobalVector<ValueType>::GlobalVector(const ParallelManager &pm) {

  LOG_DEBUG(this, "GlobalVector::GlobalVector()",
            &pm);

  assert(pm.Status() == true);

  this->object_name_ = "";

  this->pm_ = &pm;

  this->recv_boundary_ = NULL;
  this->send_boundary_ = NULL;

}

template <typename ValueType>
GlobalVector<ValueType>::~GlobalVector() {

  LOG_DEBUG(this, "GlobalVector::~GlobalVector()",
            "default destructor");

  this->Clear();

}

template <typename ValueType>
void GlobalVector<ValueType>::Clear(void) {

  LOG_DEBUG(this, "GlobalVector::Clear()",
            "");

  this->vector_interior_.Clear();
  this->vector_ghost_.Clear();

  if (this->recv_boundary_ != NULL) {
    free_host(&this->recv_boundary_);
  }

  if (this->send_boundary_ != NULL) {
    free_host(&this->send_boundary_);
  }

}

template <typename ValueType>
void GlobalVector<ValueType>::SetParallelManager(const ParallelManager &pm) {

  LOG_DEBUG(this, "GlobalVector::SetParallelManager()",
            &pm);

  assert (pm.Status() == true);

  this->pm_ = &pm;

}

template <typename ValueType>
IndexType2 GlobalVector<ValueType>::get_size(void) const {

  return this->pm_->global_size_;

}

template <typename ValueType>
int GlobalVector<ValueType>::get_local_size(void) const {

  return this->vector_interior_.get_local_size();

}

template <typename ValueType>
int GlobalVector<ValueType>::get_ghost_size(void) const {

  return this->vector_ghost_.get_local_size();

}

template <typename ValueType>
const LocalVector<ValueType>& GlobalVector<ValueType>::GetInterior() const {

  LOG_DEBUG(this, "GlobalVector::GetInterior() const",
            "");

  return this->vector_interior_;

}

template <typename ValueType>
LocalVector<ValueType>& GlobalVector<ValueType>::GetInterior() {

  LOG_DEBUG(this, "GlobalVector::GetInterior()",
            "");

  return this->vector_interior_;

}

template <typename ValueType>
const LocalVector<ValueType>& GlobalVector<ValueType>::GetGhost() const {

  LOG_DEBUG(this, "GlobalVector::GetGhost()",
            "");

  return this->vector_ghost_;

}

template <typename ValueType>
void GlobalVector<ValueType>::Allocate(std::string name, const IndexType2 size) {

  LOG_DEBUG(this, "GlobalVector::Allocate()",
            name);

  assert (this->pm_ != NULL);
  assert (this->pm_->global_size_ == size);
  assert (size <= std::numeric_limits<IndexType2>::max());

  std::string interior_name = "Interior of " + name;
  std::string ghost_name = "Ghost of " + name;

#ifdef SUPPORT_MULTINODE
  this->recv_event_ = new MRequest[this->pm_->nrecv_];
  this->send_event_ = new MRequest[this->pm_->nsend_];
#endif

  this->object_name_ = name;

  this->vector_interior_.Allocate(interior_name, this->pm_->GetLocalSize());
  this->vector_ghost_.Allocate(ghost_name, this->pm_->GetNumReceivers());

  this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

  // Allocate send and receive buffer
  allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
  allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);

}

template <typename ValueType>
void GlobalVector<ValueType>::Zeros(void) {

  LOG_DEBUG(this, "GlobalVector::Zeros()",
            "");

  this->vector_interior_.Zeros();

}

template <typename ValueType>
void GlobalVector<ValueType>::Ones(void) {

  LOG_DEBUG(this, "GlobalVector::Ones()",
            "");

  this->vector_interior_.Ones();

}

template <typename ValueType>
void GlobalVector<ValueType>::SetValues(const ValueType val) {

  LOG_DEBUG(this, "GlobalVector::SetValues()",
            val);

  this->vector_interior_.SetValues(val);

}

template <typename ValueType>
void GlobalVector<ValueType>::SetDataPtr(ValueType **ptr, std::string name, const IndexType2 size) {

  LOG_DEBUG(this, "GlobalVector::SetDataPtr()",
            name);

  assert (this->pm_ != NULL);
  assert (*ptr != NULL);
  assert (this->pm_->global_size_ == size);
  assert (size <= std::numeric_limits<IndexType2>::max());

  this->Clear();

  std::string interior_name = "Interior of " + name;
  std::string ghost_name = "Ghost of " + name;

#ifdef SUPPORT_MULTINODE
  this->recv_event_ = new MRequest[this->pm_->nrecv_];
  this->send_event_ = new MRequest[this->pm_->nsend_];
#endif

  this->object_name_ = name;

  this->vector_interior_.SetDataPtr(ptr, interior_name, this->pm_->local_size_);
  this->vector_ghost_.Allocate(ghost_name, this->pm_->GetNumReceivers());

  this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

  // Allocate send and receive buffer
  allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
  allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);

}

template <typename ValueType>
void GlobalVector<ValueType>::LeaveDataPtr(ValueType **ptr) {

  LOG_DEBUG(this, "GlobalVector::LeaveDataPtr()",
            "");

  assert(*ptr == NULL);
  assert(this->vector_interior_.get_size() > 0);

  this->vector_interior_.LeaveDataPtr(ptr);

  free_host(&this->recv_boundary_);
  free_host(&this->send_boundary_);

  this->vector_ghost_.Clear();

}

template <typename ValueType>
void GlobalVector<ValueType>::SetRandom(const ValueType a, const ValueType b, const int seed) {

  LOG_DEBUG(this, "GlobalVector::SetRandom()",
            "");

  this->vector_interior_.SetRandom(a, b, seed);

}

template <typename ValueType>
void GlobalVector<ValueType>::CopyFrom(const GlobalVector<ValueType> &src) {

  LOG_DEBUG(this, "GlobalVector::CopyFrom()",
            "");

  assert (this != &src);
  assert (this->pm_ == src.pm_);
  assert (this->recv_boundary_ != NULL);
  assert (this->send_boundary_ != NULL);

  this->vector_interior_.CopyFrom(src.vector_interior_);

}

template <typename ValueType>
void GlobalVector<ValueType>::CloneFrom(const GlobalVector<ValueType> &src) {

  LOG_DEBUG(this, "GlobalVector::CloneFrom()",
            "");

  FATAL_ERROR(__FILE__,__LINE__);

}

template <typename ValueType>
void GlobalVector<ValueType>::MoveToAccelerator(void) {

  LOG_DEBUG(this, "GlobalVector::MoveToAccelerator()",
            "");

  this->vector_interior_.MoveToAccelerator();
  this->vector_ghost_.MoveToAccelerator();

}

template <typename ValueType>
void GlobalVector<ValueType>::MoveToHost(void) {

  LOG_DEBUG(this, "GlobalVector::MoveToHost()",
            "");

  this->vector_interior_.MoveToHost();
  this->vector_ghost_.MoveToHost();

}

template <typename ValueType>
ValueType&  GlobalVector<ValueType>::operator[](const int i) {

  assert((i >= 0) && (i < this->pm_->local_size_));

  return this->vector_interior_[i];

}

template <typename ValueType>
const ValueType& GlobalVector<ValueType>::operator[](const int i) const { 

  assert((i >= 0) && (i < this->pm_->local_size_));

  return this->vector_interior_[i];

}

template <typename ValueType>
void GlobalVector<ValueType>::info(void) const {

  std::string current_backend_name;

  if (this->is_host() == true) {

    current_backend_name = _rocalution_host_name[0];

  } else {

    assert (this->is_accel() == true);
    current_backend_name = _rocalution_backend_name[this->local_backend_.backend];

  }

  LOG_INFO("GlobalVector" <<
           " name=" << this->object_name_ << ";" <<
           " size=" << this->get_size() << ";" <<
           " prec=" << 8*sizeof(ValueType) << "bit;" <<
           " subdomains=" << this->pm_->num_procs_ << ";" <<
           " host backend={" << _rocalution_host_name[0] << "};" <<
           " accelerator backend={" << _rocalution_backend_name[this->local_backend_.backend] << "};" <<
           " current=" << current_backend_name);

}

template <typename ValueType>
bool GlobalVector<ValueType>::Check(void) const {

  bool interior_check = this->vector_interior_.Check();
  bool ghost_check = this->vector_ghost_.Check();

  if (interior_check == true && ghost_check == true)
    return true;

  return false;

}

template <typename ValueType>
void GlobalVector<ValueType>::ReadFileASCII(const std::string filename) {

  LOG_DEBUG(this, "GlobalVector::ReadFileASCII()",
            filename);

  assert(this->pm_->Status() == true);

  // Read header file
  std::ifstream headfile(filename.c_str(), std::ifstream::in);

  if (!headfile.is_open()) {
    LOG_INFO("Cannot open GlobalVector file [read]: " << filename);
    FATAL_ERROR(__FILE__, __LINE__);
  }

  // Go to this ranks line in the headfile
  for (int i=0; i<this->pm_->rank_; ++i)
    headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::string name;
  std::getline(headfile, name);

  headfile.close();

  // Extract directory containing the subfiles
  size_t found = filename.find_last_of("\\/");
  std::string path = filename.substr(0, found+1);

  name.erase(remove_if(name.begin(), name.end(), isspace), name.end());

  this->vector_interior_.ReadFileASCII(path + name);

#ifdef SUPPORT_MULTINODE
  this->recv_event_ = new MRequest[this->pm_->nrecv_];
  this->send_event_ = new MRequest[this->pm_->nsend_];
#endif

  this->object_name_ = filename;

  this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

  // Allocate ghost vector
  this->vector_ghost_.Allocate("ghost", this->pm_->GetNumReceivers());

  // Allocate send and receive buffer
  allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
  allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);

}

template <typename ValueType>
void GlobalVector<ValueType>::WriteFileASCII(const std::string filename) const {

  LOG_DEBUG(this, "GlobalVector::WriteFileASCII()",
            filename);

  // Master rank writes the global headfile
  if (this->pm_->rank_ == 0) {

    std::ofstream headfile;

    headfile.open((char*)filename.c_str(), std::ofstream::out);
    if (!headfile.is_open()) {
      LOG_INFO("Cannot open GlobalVector file [write]: " << filename);
      FATAL_ERROR(__FILE__, __LINE__);
    }

    for (int i=0; i<this->pm_->num_procs_; ++i) {

      std::ostringstream rs;
      rs << i;

      std::string name = filename + ".rank." + rs.str();

      headfile << name << "\n";

    }

  }

  std::ostringstream rs;
  rs << this->pm_->rank_;

  std::string name = filename + ".rank." + rs.str();

  this->vector_interior_.WriteFileASCII(name);

}

template <typename ValueType>
void GlobalVector<ValueType>::ReadFileBinary(const std::string filename) {

  LOG_DEBUG(this, "GlobalVector::ReadFileBinary()",
            filename);

  assert(this->pm_->Status() == true);

  // Read header file
  std::ifstream headfile(filename.c_str(), std::ifstream::in);

  if (!headfile.is_open()) {
    LOG_INFO("Cannot open GlobalVector file [read]: " << filename);
    FATAL_ERROR(__FILE__, __LINE__);
  }

  // Go to this ranks line in the headfile
  for (int i=0; i<this->pm_->rank_; ++i)
    headfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::string name;
  std::getline(headfile, name);

  headfile.close();

  // Extract directory containing the subfiles
  size_t found = filename.find_last_of("\\/");
  std::string path = filename.substr(0, found+1);

  name.erase(remove_if(name.begin(), name.end(), isspace), name.end());

  this->vector_interior_.ReadFileBinary(path + name);

#ifdef SUPPORT_MULTINODE
  this->recv_event_ = new MRequest[this->pm_->nrecv_];
  this->send_event_ = new MRequest[this->pm_->nsend_];
#endif

  this->object_name_ = filename;

  this->vector_interior_.SetIndexArray(this->pm_->GetNumSenders(), this->pm_->boundary_index_);

  // Allocate ghost vector
  this->vector_ghost_.Allocate("ghost", this->pm_->GetNumReceivers());

  // Allocate send and receive buffer
  allocate_host(this->pm_->GetNumReceivers(), &this->recv_boundary_);
  allocate_host(this->pm_->GetNumSenders(), &this->send_boundary_);

}

template <typename ValueType>
void GlobalVector<ValueType>::WriteFileBinary(const std::string filename) const {

  LOG_DEBUG(this, "GlobalVector::WriteFileBinary()",
            filename);

  // Master rank writes the global headfile
  if (this->pm_->rank_ == 0) {

    std::ofstream headfile;

    headfile.open((char*)filename.c_str(), std::ofstream::out);
    if (!headfile.is_open()) {
      LOG_INFO("Cannot open GlobalVector file [write]: " << filename);
      FATAL_ERROR(__FILE__, __LINE__);
    }

    for (int i=0; i<this->pm_->num_procs_; ++i) {

      std::ostringstream rs;
      rs << i;

      std::string name = filename + ".rank." + rs.str();

      headfile << name << "\n";

    }

  }

  std::ostringstream rs;
  rs << this->pm_->rank_;

  std::string name = filename + ".rank." + rs.str();

  this->vector_interior_.WriteFileBinary(name);

}

template <typename ValueType>
void GlobalVector<ValueType>::AddScale(const GlobalVector<ValueType> &x, const ValueType alpha) {          

  LOG_DEBUG(this, "GlobalVector::Addscale()",
            "");

  this->vector_interior_.AddScale(x.vector_interior_, alpha);

}

template <typename ValueType>
void GlobalVector<ValueType>::ScaleAdd2(const ValueType alpha, const GlobalVector<ValueType> &x,
                                        const ValueType beta, const GlobalVector<ValueType> &y,
                                        const ValueType gamma) {

  LOG_DEBUG(this, "GlobalVector::ScaleAdd2()",
            "");

  this->vector_interior_.ScaleAdd2(alpha, x.vector_interior_, beta, y.vector_interior_, gamma);

}

template <typename ValueType>
void GlobalVector<ValueType>::ScaleAdd(const ValueType alpha, const GlobalVector<ValueType> &x) {

  LOG_DEBUG(this, "GlobalVector::ScaleAdd()",
            "");

  this->vector_interior_.ScaleAdd(alpha, x.vector_interior_);

}

template <typename ValueType>
void GlobalVector<ValueType>::Scale(const ValueType alpha) {

  LOG_DEBUG(this, "GlobalVector::Scale()",
            "");

  this->vector_interior_.Scale(alpha);

}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Dot(const GlobalVector<ValueType> &x) const {

  LOG_DEBUG(this, "GlobalVector::Dot()",
            "");

  ValueType local = this->vector_interior_.Dot(x.vector_interior_);
  ValueType global;

#ifdef SUPPORT_MULTINODE
  communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
  global = local;
#endif

  return global;

}

template <typename ValueType>
ValueType GlobalVector<ValueType>::DotNonConj(const GlobalVector<ValueType> &x) const {

  LOG_DEBUG(this, "GlobalVector::DotNonConj()",
            "");

  ValueType local = this->vector_interior_.DotNonConj(x.vector_interior_);
  ValueType global;

#ifdef SUPPORT_MULTINODE
  communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
  global = local;
#endif

  return global;

}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Norm(void) const {

  LOG_DEBUG(this, "GlobalVector::Norm()",
            "");

  ValueType result = this->Dot(*this);
  return sqrt(result);

}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Reduce(void) const {

  LOG_DEBUG(this, "GlobalVector::Reduce()",
            "");

  ValueType local = this->vector_interior_.Reduce();
  ValueType global;

#ifdef SUPPORT_MULTINODE
  communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
  global = local;
#endif

  return global;

}

template <typename ValueType>
ValueType GlobalVector<ValueType>::Asum(void) const {

  LOG_DEBUG(this, "GlobalVector::Asum()",
            "");

  ValueType local = this->vector_interior_.Asum();
  ValueType global;

#ifdef SUPPORT_MULTINODE
  communication_allreduce_single_sum(local, &global, this->pm_->comm_);
#else
  global = local;
#endif

  return global;

}

template <typename ValueType>
int GlobalVector<ValueType>::Amax(ValueType &value) const {

  LOG_DEBUG(this, "GlobalVector::Amax()",
            "");
  FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void GlobalVector<ValueType>::PointWiseMult(const GlobalVector<ValueType> &x) {

  LOG_DEBUG(this, "GlobalVector::PointWiseMult()",
            "");

  this->vector_interior_.PointWiseMult(x.vector_interior_);

}

template <typename ValueType>
void GlobalVector<ValueType>::PointWiseMult(const GlobalVector<ValueType> &x, const GlobalVector<ValueType> &y) {

  LOG_DEBUG(this, "GlobalVector::PointWiseMult()",
            "");

  this->vector_interior_.PointWiseMult(x.vector_interior_, y.vector_interior_);

}

template <typename ValueType>
void GlobalVector<ValueType>::UpdateGhostValuesAsync(const GlobalVector<ValueType> &in) {

  LOG_DEBUG(this, "GlobalVector::UpdateGhostValuesAsync()",
            "#*# begin");

#ifdef SUPPORT_MULTINODE
  in.vector_interior_.GetIndexValues(this->send_boundary_);

  int tag = 0;

  // async recv boundary from neighbors
  for (int i=0; i<this->pm_->nrecv_; ++i) {

    int boundary_nnz = this->pm_->recv_offset_index_[i+1] - this->pm_->recv_offset_index_[i];

    // if this has ghost values that belong to process i
    if (boundary_nnz > 0) {

      communication_async_recv(this->recv_boundary_ + this->pm_->recv_offset_index_[i],
                            boundary_nnz,
                            this->pm_->recvs_[i], 
                            tag,
                            &this->recv_event_[i],
                            this->pm_->comm_);
    }

  }

  // asyc send boundary to neighbors
  for (int i=0; i<this->pm_->nsend_; ++i) {

    int boundary_nnz = this->pm_->send_offset_index_[i+1] - this->pm_->send_offset_index_[i];

    // if process i has ghost values that belong to this
    if (boundary_nnz > 0) {

      communication_async_send(this->send_boundary_ + this->pm_->send_offset_index_[i],
                            boundary_nnz,
                            this->pm_->sends_[i],
                            tag,
                            &this->send_event_[i],
                            this->pm_->comm_);

    }

  }
#endif

  LOG_DEBUG(this, "GlobalVector::UpdateGhostValuesAsync()",
            "#*# end");

}

template <typename ValueType>
void GlobalVector<ValueType>::UpdateGhostValuesSync(void) {

  LOG_DEBUG(this, "GlobalVector::UpdateGhostValuesSync()",
            "#*# begin");

#ifdef SUPPORT_MULTINODE
  // Sync before updating ghost values
  communication_syncall(this->pm_->nrecv_, this->recv_event_);
  communication_syncall(this->pm_->nsend_, this->send_event_);

  this->vector_ghost_.SetContinuousValues(0, this->pm_->GetNumReceivers(), this->recv_boundary_);
#endif

  LOG_DEBUG(this, "GlobalVector::UpdateGhostValuesSync()",
            "#*# end");

}

template <typename ValueType>
void GlobalVector<ValueType>::Power(const double power) {

  LOG_DEBUG(this, "GlobalVector::Power()",
            "");

  this->vector_interior_.Power(power);

}

template <typename ValueType>
void GlobalVector<ValueType>::Restriction(const GlobalVector<ValueType> &vec_fine, const LocalVector<int> &map) {
}

template <typename ValueType>
void GlobalVector<ValueType>::Prolongation(const GlobalVector<ValueType> &vec_coarse, const LocalVector<int> &map) {
}

template <typename ValueType>
bool GlobalVector<ValueType>::is_host(void) const {

  assert(this->vector_interior_.is_host() == this->vector_ghost_.is_host());
  return this->vector_interior_.is_host();

}

template <typename ValueType>
bool GlobalVector<ValueType>::is_accel(void) const {

  assert(this->vector_interior_.is_accel() == this->vector_ghost_.is_accel());
  return this->vector_interior_.is_accel();

}

template class GlobalVector<double>;
template class GlobalVector<float>;
#ifdef SUPPORT_COMPLEX
template class GlobalVector<std::complex<double> >;
template class GlobalVector<std::complex<float> >;
#endif

}
