#ifndef PARALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_
#define PARALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_

#include "preconditioner.hpp"

namespace paralution {

//// BlockJacobi preconditioner for GlobalMatrix
template <class OperatorType, class VectorType, typename ValueType>
class BlockJacobi : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  BlockJacobi();
  virtual ~BlockJacobi();

  virtual void Print(void) const;
  virtual void Init(Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> &precond);

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

  virtual void SolveZeroSol(const VectorType &rhs,
                            VectorType *x);

  virtual void Build(void);
  virtual void ReBuildNumeric(void);
  virtual void Clear(void);

protected:

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void) ;

private:

  Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> *local_precond_;

};


};

#endif // PARALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_
