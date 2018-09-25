/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_DIRECT_QR_HPP_
#define ROCALUTION_DIRECT_QR_HPP_

#include "../solver.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class QR
  * \brief QR Decomposition
  * \details
  * The QR Decomposition decomposes a given matrix into \f$A = QR\f$, such that \f$Q\f$
  * is an orthogonal matrix and \f$R\f$ an upper triangular matrix.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class QR : public DirectLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    QR();
    virtual ~QR();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    virtual void Solve_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType qr_;
};

} // namespace rocalution

#endif // ROCALUTION_DIRECT_QR_HPP_
