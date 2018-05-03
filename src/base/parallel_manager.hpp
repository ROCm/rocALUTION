#ifndef ROCALUTION_PARALLEL_MANAGER_HPP_
#define ROCALUTION_PARALLEL_MANAGER_HPP_

#include "base_rocalution.hpp"
#include "../utils/types.hpp"

#include <string>
#include <complex>

namespace rocalution {

template <typename ValueType>
class GlobalMatrix;
template <typename ValueType>
class GlobalVector;

// ParallelManager
class ParallelManager : public RocalutionObj {

public:

  ParallelManager();
  ~ParallelManager();

  // ALL set functions must be called only once
  void SetMPICommunicator(const void *comm);
  void Clear(void);

  IndexType2 GetGlobalSize(void) const;
  int GetLocalSize(void) const;

  int GetNumReceivers(void) const;
  int GetNumSenders(void) const;
  int GetNumProcs(void) const;

  void SetGlobalSize(const IndexType2 size);
  void SetLocalSize(const int size);

  // Contains all boundary indices of current process -- local or global indices?
  void SetBoundaryIndex(const int size, const int *index);

  // Number of receivers, array of the receiver process number, offsets where the boundary for process 'receiver' starts
  void SetReceivers(const int nrecv, const int *recvs, const int *recv_offset);
  // Number of senders, array of the sender process number, offsets where the ghost for process 'sender' starts
  void SetSenders(const int nsend, const int *sends, const int *send_offset);

  // Mapping local to global and global to local
  void LocalToGlobal(const int proc, const int local, int &global);
  void GlobalToLocal(const int global, int &proc, int &local);

  bool Status(void) const;

  // Read file that contains all relevant PM data
  void ReadFileASCII(const std::string filename);
  // Write file that contains all relevant PM data
  void WriteFileASCII(const std::string filename) const;

private:

  const void *comm_;
  int rank_;
  int num_procs_;

  IndexType2 global_size_;
  int local_size_;

  int recv_index_size_;
  int send_index_size_;

  int nrecv_;
  int nsend_;

  int *recvs_;
  int *sends_;

  int *recv_offset_index_;
  int *send_offset_index_;

  int *boundary_index_;

  friend class GlobalMatrix<double>;
  friend class GlobalMatrix<float>;
  friend class GlobalMatrix<std::complex<double> >;
  friend class GlobalMatrix<std::complex<float> >;
  friend class GlobalVector<double>;
  friend class GlobalVector<float>;
  friend class GlobalVector<std::complex<double> >;
  friend class GlobalVector<std::complex<float> >;
  friend class GlobalVector<int>;

};

}

#endif // ROCALUTION_PARALLEL_MANAGER_HPP_
