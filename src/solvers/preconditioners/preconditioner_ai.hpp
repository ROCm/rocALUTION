#ifndef PARALUTION_PRECONDITIONER_AI_HPP_
#define PARALUTION_PRECONDITIONER_AI_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"

namespace paralution {

/// Approximate Inverse - Chebyshev preconditioner
/// see IEEE TRANSACTIONS ON POWER SYSTEMS, VOL. 18, NO. 4, NOVEMBER 2003;
/// A New Preconditioned Conjugate Gradient Power Flow -
/// Hasan Dag, Adam Semlyen
template <class OperatorType, class VectorType, typename ValueType>
class AIChebyshev : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  AIChebyshev();
  virtual ~AIChebyshev();

  virtual void Print(void) const;  
  virtual void Solve(const VectorType &rhs,
                     VectorType *x);
  virtual void Set(const int p, const ValueType lambda_min, const ValueType lambda_max);
  virtual void Build(void);
  virtual void Clear(void);  


protected:

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);


private:

  OperatorType AIChebyshev_;
  int p_;
  ValueType lambda_min_, lambda_max_;

};

/// Factorized Approximate Inverse preconditioner
template <class OperatorType, class VectorType, typename ValueType>
class FSAI : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  FSAI();
  virtual ~FSAI();

  virtual void Print(void) const;
  virtual void Solve(const VectorType &rhs, VectorType *x);
  /// Initialize the FSAI with powered system matrix sparsity pattern
  virtual void Set(const int power);
  /// Initialize the FSAI with external sparsity pattern
  virtual void Set(const OperatorType &pattern);
  virtual void Build(void);
  virtual void Clear(void);

  virtual void SetPrecondMatrixFormat(const unsigned int mat_format);

protected:

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

private:

  OperatorType FSAI_L_;
  OperatorType FSAI_LT_;
  VectorType t_;

  int matrix_power_;

  bool external_pattern_;
  const OperatorType *matrix_pattern_;

  /// Keep the precond matrix in CSR or not
  bool op_mat_format_;
  /// Precond matrix format
  unsigned int precond_mat_format_;

};

/// SParse Approximate Inverse preconditioner
template <class OperatorType, class VectorType, typename ValueType>
class SPAI : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  SPAI();
  virtual ~SPAI();

  virtual void Print(void) const;  
  virtual void Solve(const VectorType &rhs, VectorType *x);
  virtual void Build(void);
  virtual void Clear(void);

  virtual void SetPrecondMatrixFormat(const unsigned int mat_format);


protected:

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);


private:

  OperatorType SPAI_;

  /// Keep the precond matrix in CSR or not
  bool op_mat_format_; 
  /// Precond matrix format
  unsigned int precond_mat_format_;

};


/// Truncated Neumann Series (TNS) Preconditioner
template <class OperatorType, class VectorType, typename ValueType>
class TNS : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  TNS();
  virtual ~TNS();

  virtual void Print(void) const;  
  virtual void Set(const bool imp);
  virtual void Solve(const VectorType &rhs, VectorType *x);
  virtual void Build(void);
  virtual void Clear(void);

  virtual void SetPrecondMatrixFormat(const unsigned int mat_format);


protected:

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);


private:

  OperatorType L_;
  OperatorType LT_;
  OperatorType TNS_;
  VectorType Dinv_;

  VectorType tmp1_;
  VectorType tmp2_;

  /// Keep the precond matrix in CSR or not
  bool op_mat_format_; 
  /// Precond matrix format
  unsigned int precond_mat_format_;
  /// implicit (true) or explicit (false) computation 
  bool impl_;

};


}

#endif // PARALUTION_PRECONDITIONER_AI_HPP_
