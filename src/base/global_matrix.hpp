#ifndef PARALUTION_GLOBAL_MATRIX_HPP_
#define PARALUTION_GLOBAL_MATRIX_HPP_

#include "../utils/types.hpp"
#include "operator.hpp"
#include "parallel_manager.hpp"

namespace paralution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class LocalVector;

template <typename ValueType>
class LocalMatrix;

// Global Matrix
template <typename ValueType>
class GlobalMatrix : public Operator<ValueType> {

public:

  GlobalMatrix();
  GlobalMatrix(const ParallelManager &pm);
  virtual ~GlobalMatrix();

  virtual IndexType2 get_nrow(void) const;
  virtual IndexType2 get_ncol(void) const;
  virtual IndexType2 get_nnz(void) const;
  virtual int get_local_nrow(void) const;
  virtual int get_local_ncol(void) const;
  virtual int get_local_nnz(void) const;
  virtual int get_ghost_nrow(void) const;
  virtual int get_ghost_ncol(void) const;
  virtual int get_ghost_nnz(void) const;

  // TODO
  // do we need this to be public?
  // can we have the GetGhost in the protected section?
  const LocalMatrix<ValueType>& GetInterior() const;
  const LocalMatrix<ValueType>& GetGhost() const;

  virtual void MoveToAccelerator(void); // check for ghost local and boundary parts!
  virtual void MoveToHost(void);

  virtual void info(void) const;
  virtual bool Check(void) const;

  // TODO
  // can we have all Allocation/SetData function to be protected or private?
  void AllocateCSR(std::string name, const int local_nnz, const int ghost_nnz);
  void AllocateCOO(std::string name, const int local_nnz, const int ghost_nnz);
  virtual void Clear(void);
  void SetParallelManager(const ParallelManager &pm);

  void SetDataPtrCSR(int **local_row_offset, int **local_col, ValueType **local_val,
                     int **ghost_row_offset, int **ghost_col, ValueType **ghost_val,
                     std::string name, const int local_nnz, const int ghost_nnz);
  void SetDataPtrCOO(int **local_row, int **local_col, ValueType **local_val,
                     int **ghost_row, int **ghost_col, ValueType **ghost_val,
                     std::string name, const int local_nnz, const int ghost_nnz);
  void SetLocalDataPtrCSR(int **row_offset, int **col, ValueType **val,
                          std::string name, const int nnz);
  void SetLocalDataPtrCOO(int **row, int **col, ValueType **val,
                          std::string name, const int nnz);
  void SetGhostDataPtrCSR(int **row_offset, int **col, ValueType **val,
                          std::string name, const int nnz);
  void SetGhostDataPtrCOO(int **row, int **col, ValueType **val,
                          std::string name, const int nnz);

  void LeaveDataPtrCSR(int **local_row_offset, int **local_col, ValueType **local_val,
                       int **ghost_row_offset, int **ghost_col, ValueType **ghost_val);
  void LeaveDataPtrCOO(int **local_row, int **local_col, ValueType **local_val,
                       int **ghost_row, int **ghost_col, ValueType **ghost_val);
  void LeaveLocalDataPtrCSR(int **row_offset, int **col, ValueType **val);
  void LeaveLocalDataPtrCOO(int **row, int **col, ValueType **val);
  void LeaveGhostDataPtrCSR(int **row_offset, int **col, ValueType **val);
  void LeaveGhostDataPtrCOO(int **row, int **col, ValueType **val);

  void CloneFrom(const GlobalMatrix<ValueType> &src);
  void CopyFrom(const GlobalMatrix<ValueType> &src);

  /// Convert the matrix to CSR structure
  void ConvertToCSR(void);
  /// Convert the matrix to MCSR structure
  void ConvertToMCSR(void);
  /// Convert the matrix to BCSR structure
  void ConvertToBCSR(void);
  /// Convert the matrix to COO structure
  void ConvertToCOO(void);
  /// Convert the matrix to ELL structure
  void ConvertToELL(void);
  /// Convert the matrix to DIA structure
  void ConvertToDIA(void);
  /// Convert the matrix to HYB structure
  void ConvertToHYB(void);
  /// Convert the matrix to DENSE structure
  void ConvertToDENSE(void);
  /// Convert the matrix to specified matrix ID format
  void ConvertTo(const unsigned int matrix_format);

  virtual void Apply(const GlobalVector<ValueType> &in, GlobalVector<ValueType> *out) const; 
  virtual void ApplyAdd(const GlobalVector<ValueType> &in, const ValueType scalar, 
                        GlobalVector<ValueType> *out) const;

  /// Read matrix from MTX (Matrix Market Format) file
  void ReadFileMTX(const std::string filename);
  /// Write matrix to MTX (Matrix Market Format) file
  void WriteFileMTX(const std::string filename) const;
  /// Read matrix from CSR (PARALUTION binary format) file
  void ReadFileCSR(const std::string filename);
  /// Write matrix to CSR (PARALUTION binary format) file
  void WriteFileCSR(const std::string filename) const;

  /// Sort the matrix indices
  void Sort(void);

  /// Extract the inverse (reciprocal) diagonal values of the matrix into a LocalVector
  void ExtractInverseDiagonal(GlobalVector<ValueType> *vec_inv_diag) const;

  /// Scale all the values in the matrix
  void Scale(const ValueType alpha);

  /// Initial Pairwise Aggregation scheme
  void InitialPairwiseAggregation(const ValueType beta, int &nc, LocalVector<int> *G, int &Gsize,
                                  int **rG, int &rGsize, const int ordering) const;
  /// Further Pairwise Aggregation scheme
  void FurtherPairwiseAggregation(const ValueType beta, int &nc, LocalVector<int> *G, int &Gsize,
                                  int **rG, int &rGsize, const int ordering) const;
  /// Build coarse operator for pairwise aggregation scheme
  void CoarsenOperator(GlobalMatrix<ValueType> *Ac, ParallelManager *pm, const int nrow, const int ncol,
                       const LocalVector<int> &G, const int Gsize, const int *rG, const int rGsize) const;

protected:

  virtual bool is_host(void) const;
  virtual bool is_accel(void) const;

private:

  IndexType2 nnz_;

  LocalMatrix<ValueType> matrix_interior_;
  LocalMatrix<ValueType> matrix_ghost_;

  friend class GlobalVector<ValueType>;
  friend class LocalMatrix<ValueType>;
  friend class LocalVector<ValueType>;

};


}

#endif // PARALUTION_GLOBAL_MATRIX_HPP_
