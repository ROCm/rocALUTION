#ifndef PARALUTION_DIRECT_QR_HPP_
#define PARALUTION_DIRECT_QR_HPP_

#include "../solver.hpp"

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
class QR : public DirectLinearSolver<OperatorType, VectorType, ValueType> {

public:

  QR();
  virtual ~QR();

  virtual void Print(void) const;

  virtual void Build(void);
  virtual void Clear(void);

protected:

  virtual void Solve_(const VectorType &rhs, VectorType *x);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

private:

  OperatorType qr_;

};


}

#endif // PARAQRTION_DIRECT_QR_HPP_
